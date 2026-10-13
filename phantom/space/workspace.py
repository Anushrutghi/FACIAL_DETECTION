"""PHANTOM 3D Spatial Computing Workspace Engine.

Manages 3D virtual objects, one/two-hand 6-DOF manipulation, 3D spatial drawing,
floating spatial action menus, depth ordering, and guided onboarding tutorial.
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from phantom.space.math3d import project_point_3d, unproject_screen_to_ray, ray_sphere_intersect
from phantom.space.objects import SpatialObject, SpatialCube, SpatialPyramid, SpatialHoloRing


CYAN = (255, 220, 0)
NEON_GREEN = (0, 255, 128)
AMBER = (0, 165, 255)
WHITE = (255, 255, 255)


class SpatialWorkspace:
    """Integrated 3D interactive workspace."""

    def __init__(self, width: int = 640, height: int = 480, focal_length: float = 600.0) -> None:
        self.width = width
        self.height = height
        self.focal_length = focal_length

        # Scene objects
        self.objects: List[SpatialObject] = [
            SpatialCube(name="CUBE-ALPHA", pos=(-110.0, 0.0, 500.0), scale=75.0, color=CYAN),
            SpatialPyramid(name="PYRAMID-BETA", pos=(110.0, 0.0, 520.0), scale=70.0, color=(255, 0, 220)),
            SpatialHoloRing(name="GYRO-CORE", pos=(0.0, 90.0, 560.0), scale=65.0, color=NEON_GREEN),
        ]

        # Spatial 3D Drawing Strokes: list of list of (x, y, z, bgr)
        self.spatial_strokes: List[List[Tuple[float, float, float, Tuple[int, int, int]]]] = []
        self._current_stroke: List[Tuple[float, float, float, Tuple[int, int, int]]] = []
        self.spatial_draw_active = False

        # Guided Interactive Demo Tutorial State
        self.tutorial_active = True
        self.tutorial_step = 1  # 1 to 4
        self.tutorial_step_timer = time.time()
        self.tutorial_completed_steps = set()

        # Floating 3D Menu Buttons: (id, name, 3D pos, radius, color)
        self.menu_buttons = [
            {"id": "btn_cube", "label": "+ CUBE", "pos": np.array([-160.0, -130.0, 480.0], dtype=np.float32), "r": 28.0, "color": CYAN},
            {"id": "btn_pyr", "label": "+ PYR", "pos": np.array([-80.0, -130.0, 480.0], dtype=np.float32), "r": 28.0, "color": (255, 0, 220)},
            {"id": "btn_reset", "label": "RESET", "pos": np.array([0.0, -130.0, 480.0], dtype=np.float32), "r": 28.0, "color": AMBER},
            {"id": "btn_draw3d", "label": "3D DRAW", "pos": np.array([80.0, -130.0, 480.0], dtype=np.float32), "r": 28.0, "color": NEON_GREEN},
            {"id": "btn_tuto", "label": "GUIDE", "pos": np.array([160.0, -130.0, 480.0], dtype=np.float32), "r": 28.0, "color": WHITE},
        ]
        self._btn_cooldown = 0.4
        self._last_btn_time = 0.0

    def reset_objects(self) -> None:
        self.objects = [
            SpatialCube(name="CUBE-ALPHA", pos=(-110.0, 0.0, 500.0), scale=75.0, color=CYAN),
            SpatialPyramid(name="PYRAMID-BETA", pos=(110.0, 0.0, 520.0), scale=70.0, color=(255, 0, 220)),
            SpatialHoloRing(name="GYRO-CORE", pos=(0.0, 90.0, 560.0), scale=65.0, color=NEON_GREEN),
        ]
        self.spatial_strokes.clear()
        self._current_stroke.clear()

    def update_and_interact(
        self,
        hand_states: List[Dict[str, Any]],
        dt: float = 0.033,
    ) -> None:
        """Processes hand positions, pinch actions, 3D selection, and manipulation."""
        # 1. Idle rotation for visual feedback
        for obj in self.objects:
            if not obj.is_grabbed:
                obj.rot[1] += dt * 0.4  # Gentle idle yaw

        # Reset hover flags
        for obj in self.objects:
            obj.is_hovered = False

        if not hand_states:
            # Release any held objects
            for obj in self.objects:
                obj.release_grab()
            self._end_current_3d_stroke()
            return

        # 2. Check 2-hand dual grab first if two hands are detected
        if len(hand_states) >= 2:
            h0, h1 = hand_states[0], hand_states[1]
            p0, p1 = h0["index_tip"], h1["index_tip"]
            both_pinch = h0["pinch"] and h1["pinch"]

            # Check if an object is already grabbed by both hands
            dual_grabbed_obj = next((o for o in self.objects if o.is_grabbed and len(o.grabbed_by) == 2), None)

            if both_pinch:
                if dual_grabbed_obj:
                    dual_grabbed_obj.update_dual_grab(p0, p1, self.width, self.height, self.focal_length)
                    if self.tutorial_active and self.tutorial_step == 4:
                        self.tutorial_step = 5  # Completed
                else:
                    # Check if midpoint is near any object
                    mid_x, mid_y = (p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2
                    target = self._find_object_at_screen_pos((mid_x, mid_y))
                    if target:
                        target.start_dual_grab(p0, p1)
                        if self.tutorial_active and self.tutorial_step == 4:
                            self.tutorial_step = 5
            else:
                if dual_grabbed_obj:
                    dual_grabbed_obj.release_grab()

        # 3. Single-hand selection & manipulation
        for state in hand_states:
            idx = state["hand_index"]
            tip = state["index_tip"]
            pinch = state["pinch"]
            palm_scale = state.get("palm_scale", 60.0)
            palm_ratio = max(0.4, min(2.5, palm_scale / 60.0))

            # Find object grabbed by this hand
            grabbed_obj = next((o for o in self.objects if o.is_grabbed and idx in o.grabbed_by and len(o.grabbed_by) == 1), None)

            if pinch:
                if grabbed_obj:
                    grabbed_obj.update_single_grab(idx, tip, palm_ratio, self.width, self.height, self.focal_length)
                    if self.tutorial_active and self.tutorial_step in (2, 3):
                        self.tutorial_step = max(self.tutorial_step, 3)
                else:
                    # Check if hovering over an object to grab
                    target = self._find_object_at_screen_pos(tip)
                    if target and not target.is_grabbed:
                        target.start_single_grab(idx, tip)
                        if self.tutorial_active and self.tutorial_step == 2:
                            self.tutorial_step = 3
                    else:
                        # Check 3D menu button click
                        self._check_menu_button_click(tip)
            else:
                # Release grab for this hand
                if grabbed_obj:
                    grabbed_obj.release_grab(idx)

                # Update hover feedback
                hovered = self._find_object_at_screen_pos(tip)
                if hovered:
                    hovered.is_hovered = True
                    if self.tutorial_active and self.tutorial_step == 1:
                        self.tutorial_step = 2

                # 4. Handle 3D Spatial Drawing if enabled
                if self.spatial_draw_active and state["index_up"] and not state["palm"]:
                    # Compute 3D world point at hand depth
                    depth = 500.0 / palm_ratio
                    factor = depth / self.focal_length
                    wx = (tip[0] - self.width / 2.0) * factor
                    wy = -(tip[1] - self.height / 2.0) * factor
                    wz = depth
                    color = CYAN if idx == 0 else (255, 0, 220)
                    self._current_stroke.append((wx, wy, wz, color))
                else:
                    self._end_current_3d_stroke()

    def _end_current_3d_stroke(self) -> None:
        if len(self._current_stroke) > 1:
            self.spatial_strokes.append(list(self._current_stroke))
        self._current_stroke.clear()

    def _find_object_at_screen_pos(self, screen_pos: Tuple[int, int]) -> Optional[SpatialObject]:
        """Finds closest object whose projected 2D bounds enclose the given point."""
        closest_obj = None
        min_depth = 1e9

        for obj in self.objects:
            proj = project_point_3d(obj.pos, self.width, self.height, self.focal_length)
            if proj is None:
                continue

            cx, cy, depth = proj
            scr_radius = (obj.bounding_radius * self.focal_length) / depth

            dist = math.hypot(screen_pos[0] - cx, screen_pos[1] - cy)
            if dist <= scr_radius * 1.2 and depth < min_depth:
                min_depth = depth
                closest_obj = obj

        return closest_obj

    def _check_menu_button_click(self, screen_pos: Tuple[int, int]) -> None:
        now = time.time()
        if now - self._last_btn_time < self._btn_cooldown:
            return

        for btn in self.menu_buttons:
            proj = project_point_3d(btn["pos"], self.width, self.height, self.focal_length)
            if proj is None:
                continue
            cx, cy, depth = proj
            r_px = (btn["r"] * self.focal_length) / depth

            if math.hypot(screen_pos[0] - cx, screen_pos[1] - cy) <= r_px:
                self._trigger_button_action(btn["id"])
                self._last_btn_time = now
                break

    def _trigger_button_action(self, btn_id: str) -> None:
        if btn_id == "btn_cube":
            cnt = len([o for o in self.objects if isinstance(o, SpatialCube)]) + 1
            self.objects.append(SpatialCube(name=f"CUBE-{cnt}", pos=(0.0, 20.0, 520.0), scale=65.0, color=CYAN))
        elif btn_id == "btn_pyr":
            cnt = len([o for o in self.objects if isinstance(o, SpatialPyramid)]) + 1
            self.objects.append(SpatialPyramid(name=f"PYR-{cnt}", pos=(0.0, 20.0, 520.0), scale=60.0, color=(255, 0, 220)))
        elif btn_id == "btn_reset":
            self.reset_objects()
        elif btn_id == "btn_draw3d":
            self.spatial_draw_active = not self.spatial_draw_active
        elif btn_id == "btn_tuto":
            self.tutorial_active = not self.tutorial_active
            self.tutorial_step = 1

    def render(self, frame: np.ndarray) -> None:
        """Renders 3D horizon grid, spatial strokes, virtual objects, and 3D floating buttons."""
        self._render_perspective_grid(frame)
        self._render_spatial_strokes(frame)

        # Depth-sort objects from farthest to closest (Painter's algorithm)
        sorted_objs = sorted(self.objects, key=lambda o: o.pos[2], reverse=True)
        for obj in sorted_objs:
            obj.render(frame, self.width, self.height, self.focal_length)

        self._render_menu_buttons(frame)

        if self.tutorial_active:
            self._render_tutorial_hud(frame)

    def _render_perspective_grid(self, frame: np.ndarray) -> None:
        """Draws subtle sci-fi horizon perspective grid on the floor."""
        floor_y = -180.0
        # Longitudinal lines
        for x in range(-400, 450, 100):
            p1 = project_point_3d((float(x), floor_y, 250.0), self.width, self.height, self.focal_length)
            p2 = project_point_3d((float(x), floor_y, 900.0), self.width, self.height, self.focal_length)
            if p1 and p2:
                cv2.line(frame, (p1[0], p1[1]), (p2[0], p2[1]), (45, 60, 75), 1, cv2.LINE_AA)

        # Transverse lines
        for z in range(250, 950, 120):
            p1 = project_point_3d((-400.0, floor_y, float(z)), self.width, self.height, self.focal_length)
            p2 = project_point_3d((400.0, floor_y, float(z)), self.width, self.height, self.focal_length)
            if p1 and p2:
                cv2.line(frame, (p1[0], p1[1]), (p2[0], p2[1]), (45, 60, 75), 1, cv2.LINE_AA)

    def _render_spatial_strokes(self, frame: np.ndarray) -> None:
        """Projects and renders 3D spatial drawing ribbon segments."""
        all_strokes = self.spatial_strokes + ([self._current_stroke] if self._current_stroke else [])
        for stroke in all_strokes:
            for i in range(1, len(stroke)):
                p1_3d, col1 = stroke[i - 1][:3], stroke[i - 1][3]
                p2_3d = stroke[i][:3]
                p1_scr = project_point_3d(p1_3d, self.width, self.height, self.focal_length)
                p2_scr = project_point_3d(p2_3d, self.width, self.height, self.focal_length)
                if p1_scr and p2_scr:
                    cv2.line(frame, (p1_scr[0], p1_scr[1]), (p2_scr[0], p2_scr[1]), col1, 2, cv2.LINE_AA)

    def _render_menu_buttons(self, frame: np.ndarray) -> None:
        """Renders floating 3D holographic menu buttons."""
        for btn in self.menu_buttons:
            proj = project_point_3d(btn["pos"], self.width, self.height, self.focal_length)
            if proj is None:
                continue
            cx, cy, depth = proj
            r_px = int((btn["r"] * self.focal_length) / depth)

            # Circular glow badge
            cv2.circle(frame, (cx, cy), r_px, btn["color"], 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), r_px - 3, (20, 25, 30), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), r_px - 3, btn["color"], 1, cv2.LINE_AA)

            # Label
            label = btn["label"]
            tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            cv2.putText(frame, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)

    def _render_tutorial_hud(self, frame: np.ndarray) -> None:
        """Renders interactive step-by-step guided tutorial prompt."""
        steps = {
            1: ("STEP 1/4: SPATIAL RETICLE", "Aim index fingertip to hover over the holographic Cube."),
            2: ("STEP 2/4: PINCH TO GRAB", "Pinch thumb and index together to grab the object."),
            3: ("STEP 3/4: 3D TRANSLATION", "Move your hand around and in/out to position in 3D space."),
            4: ("STEP 4/4: DUAL-HAND 6-DOF", "Pinch with BOTH hands to rotate and scale the object!"),
            5: ("TUTORIAL COMPLETE!", "Freely manipulate 3D objects or click '3D DRAW' for spatial art."),
        }

        title, desc = steps.get(self.tutorial_step, ("TUTORIAL", "Follow guided steps"))

        # Banner at top center
        bw, bh = 420, 52
        bx = (self.width - bw) // 2
        by = 14

        scrim = frame[by:by + bh, bx:bx + bw]
        tint = np.full_like(scrim, (20, 24, 30))
        cv2.addWeighted(tint, 0.8, scrim, 0.2, 0, scrim)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), NEON_GREEN, 1, cv2.LINE_AA)

        cv2.putText(frame, title, (bx + 16, by + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, NEON_GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, desc, (bx + 16, by + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)
