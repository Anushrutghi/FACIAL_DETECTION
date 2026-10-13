"""PHANTOM Interactive 3D Virtual Objects.

Holographic 3D spatial primitives (Cubes, Pyramids, Rings, Panels) supporting
single-hand 3D translation & rotation, and dual-hand scaling & 6-DOF manipulation.
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from phantom.space.math3d import project_point_3d, rotation_matrix_xyz, unproject_screen_to_ray


CYAN = (255, 220, 0)
NEON_GREEN = (0, 255, 128)
AMBER = (0, 165, 255)
MAGENTA = (255, 0, 220)
WHITE = (255, 255, 255)


class SpatialObject:
    """Base class for all interactive 3D spatial objects."""

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0.0, 0.0, 500.0),
        scale: float = 80.0,
        color: Tuple[int, int, int] = CYAN,
    ) -> None:
        self.name = name
        self.pos = np.array(pos, dtype=np.float32)
        self.rot = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Pitch, Yaw, Roll
        self.scale = float(scale)
        self.color = color

        # Interaction state
        self.is_hovered = False
        self.is_grabbed = False
        self.grabbed_by: List[int] = []

        # Grab tracking caches
        self._initial_grab_pos = np.copy(self.pos)
        self._grab_screen_anchor: Dict[int, Tuple[int, int]] = {}
        self._dual_initial_dist: float = 1.0
        self._dual_initial_scale: float = self.scale
        self._dual_initial_angle: float = 0.0

        # Geometry definition (override in subclasses)
        self.vertices_local: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.edges: List[Tuple[int, int]] = []
        self.faces: List[List[int]] = []

    @property
    def bounding_radius(self) -> float:
        return self.scale * 1.3

    def get_world_vertices(self) -> np.ndarray:
        """Transforms local vertices by scale, rotation, and translation."""
        r_mat = rotation_matrix_xyz(self.rot[0], self.rot[1], self.rot[2])
        scaled = self.vertices_local * self.scale
        rotated = (r_mat @ scaled.T).T
        return rotated + self.pos

    def start_single_grab(self, hand_idx: int, screen_pos: Tuple[int, int]) -> None:
        self.is_grabbed = True
        if hand_idx not in self.grabbed_by:
            self.grabbed_by.append(hand_idx)
        self._grab_screen_anchor[hand_idx] = screen_pos
        self._initial_grab_pos = np.copy(self.pos)

    def update_single_grab(
        self,
        hand_idx: int,
        curr_screen_pos: Tuple[int, int],
        palm_scale_ratio: float,
        width: int,
        height: int,
        focal_length: float = 600.0,
    ) -> None:
        """Translates the object in 3D based on screen delta and hand depth ratio."""
        if hand_idx not in self._grab_screen_anchor:
            self._grab_screen_anchor[hand_idx] = curr_screen_pos
            return

        anchor = self._grab_screen_anchor[hand_idx]
        dx = curr_screen_pos[0] - anchor[0]
        dy = curr_screen_pos[1] - anchor[1]

        # Convert screen delta to 3D world units at object depth
        depth = self.pos[2]
        factor = depth / focal_length

        self.pos[0] = self._initial_grab_pos[0] + dx * factor
        self.pos[1] = self._initial_grab_pos[1] - dy * factor  # Invert Y

        # Adjust depth (Z) based on palm scale changes (moving hand closer/further)
        depth_offset = (1.0 - palm_scale_ratio) * 150.0
        self.pos[2] = max(150.0, min(1200.0, self._initial_grab_pos[2] + depth_offset))

        # Gentle rotation following movement
        self.rot[1] += dx * 0.005
        self.rot[0] += dy * 0.005

    def start_dual_grab(
        self,
        h0_pos: Tuple[int, int],
        h1_pos: Tuple[int, int],
    ) -> None:
        """Initializes 2-hand grab for simultaneous scaling and rotation."""
        self.is_grabbed = True
        self.grabbed_by = [0, 1]
        self._initial_grab_pos = np.copy(self.pos)
        self._dual_initial_dist = max(10.0, math.hypot(h0_pos[0] - h1_pos[0], h0_pos[1] - h1_pos[1]))
        self._dual_initial_scale = self.scale
        self._dual_initial_angle = math.atan2(h1_pos[1] - h0_pos[1], h1_pos[0] - h0_pos[0])

    def update_dual_grab(
        self,
        h0_pos: Tuple[int, int],
        h1_pos: Tuple[int, int],
        width: int,
        height: int,
        focal_length: float = 600.0,
    ) -> None:
        """Performs two-hand 6-DOF manipulation (Midpoint translation, Distance scale, Angle rotation)."""
        curr_dist = max(10.0, math.hypot(h0_pos[0] - h1_pos[0], h0_pos[1] - h1_pos[1]))
        curr_angle = math.atan2(h1_pos[1] - h0_pos[1], h1_pos[0] - h0_pos[0])

        # 1. Scale object based on pinch distance ratio
        scale_ratio = curr_dist / self._dual_initial_dist
        self.scale = max(25.0, min(250.0, self._dual_initial_scale * scale_ratio))

        # 2. Rotate around Z axis based on angle change between hands
        d_angle = curr_angle - self._dual_initial_angle
        self.rot[2] += d_angle * 0.5
        self._dual_initial_angle = curr_angle

        # 3. Translate midpoint
        mid_x = (h0_pos[0] + h1_pos[0]) / 2.0
        mid_y = (h0_pos[1] + h1_pos[1]) / 2.0

        depth = self.pos[2]
        factor = depth / focal_length
        self.pos[0] = (mid_x - width / 2.0) * factor
        self.pos[1] = -(mid_y - height / 2.0) * factor

    def release_grab(self, hand_idx: Optional[int] = None) -> None:
        if hand_idx is not None and hand_idx in self.grabbed_by:
            self.grabbed_by.remove(hand_idx)
            self._grab_screen_anchor.pop(hand_idx, None)
        else:
            self.grabbed_by.clear()
            self._grab_screen_anchor.clear()

        if not self.grabbed_by:
            self.is_grabbed = False

    def render(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        focal_length: float = 600.0,
    ) -> None:
        """Projects world vertices and draws holographic geometry."""
        world_pts = self.get_world_vertices()
        proj_pts: List[Optional[Tuple[int, int, float]]] = [
            project_point_3d(pt, width, height, focal_length) for pt in world_pts
        ]

        # Draw wireframe edges
        active_color = AMBER if self.is_grabbed else (NEON_GREEN if self.is_hovered else self.color)
        thickness = 2 if (self.is_grabbed or self.is_hovered) else 1

        for p1_idx, p2_idx in self.edges:
            pt1 = proj_pts[p1_idx]
            pt2 = proj_pts[p2_idx]
            if pt1 is not None and pt2 is not None:
                cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), active_color, thickness, cv2.LINE_AA)

        # Draw vertices as glowing nodes
        for pt in proj_pts:
            if pt is not None:
                cv2.circle(frame, (pt[0], pt[1]), 3, WHITE, -1, cv2.LINE_AA)

        # Draw futuristic selection bounding brackets if hovered or grabbed
        center_proj = project_point_3d(self.pos, width, height, focal_length)
        if center_proj is not None and (self.is_hovered or self.is_grabbed):
            cx, cy = center_proj[0], center_proj[1]
            box_r = int((self.bounding_radius * focal_length) / max(50.0, self.pos[2]))
            b_color = AMBER if self.is_grabbed else NEON_GREEN

            # Corner brackets
            d = max(10, box_r // 3)
            # Top-left
            cv2.line(frame, (cx - box_r, cy - box_r), (cx - box_r + d, cy - box_r), b_color, 2, cv2.LINE_AA)
            cv2.line(frame, (cx - box_r, cy - box_r), (cx - box_r, cy - box_r + d), b_color, 2, cv2.LINE_AA)
            # Top-right
            cv2.line(frame, (cx + box_r, cy - box_r), (cx + box_r - d, cy - box_r), b_color, 2, cv2.LINE_AA)
            cv2.line(frame, (cx + box_r, cy - box_r), (cx + box_r, cy - box_r + d), b_color, 2, cv2.LINE_AA)
            # Bottom-left
            cv2.line(frame, (cx - box_r, cy + box_r), (cx - box_r + d, cy + box_r), b_color, 2, cv2.LINE_AA)
            cv2.line(frame, (cx - box_r, cy + box_r), (cx - box_r, cy + box_r - d), b_color, 2, cv2.LINE_AA)
            # Bottom-right
            cv2.line(frame, (cx + box_r, cy + box_r), (cx + box_r - d, cy + box_r), b_color, 2, cv2.LINE_AA)
            cv2.line(frame, (cx + box_r, cy + box_r), (cx + box_r, cy + box_r - d), b_color, 2, cv2.LINE_AA)

            # Name and depth badge
            badge_text = f"{self.name} [Z:{int(self.pos[2])}mm]"
            cv2.putText(frame, badge_text, (cx - box_r, cy - box_r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, b_color, 1, cv2.LINE_AA)


class SpatialCube(SpatialObject):
    """Holographic 3D Cube with 8 vertices and 12 edges."""

    def __init__(
        self,
        name: str = "HOLO-CUBE",
        pos: Tuple[float, float, float] = (0.0, 0.0, 500.0),
        scale: float = 80.0,
        color: Tuple[int, int, int] = CYAN,
    ) -> None:
        super().__init__(name=name, pos=pos, scale=scale, color=color)

        # Unit cube vertices centered at origin [-0.5, 0.5]
        self.vertices_local = np.array([
            [-0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5],
            [ 0.5,  0.5, -0.5],
            [-0.5,  0.5, -0.5],
            [-0.5, -0.5,  0.5],
            [ 0.5, -0.5,  0.5],
            [ 0.5,  0.5,  0.5],
            [-0.5,  0.5,  0.5],
        ], dtype=np.float32)

        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Connecting edges
        ]


class SpatialPyramid(SpatialObject):
    """Holographic 3D Pyramid with square base."""

    def __init__(
        self,
        name: str = "HOLO-PYRAMID",
        pos: Tuple[float, float, float] = (140.0, 0.0, 550.0),
        scale: float = 75.0,
        color: Tuple[int, int, int] = MAGENTA,
    ) -> None:
        super().__init__(name=name, pos=pos, scale=scale, color=color)

        self.vertices_local = np.array([
            [-0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5],
            [ 0.5, -0.5,  0.5],
            [-0.5, -0.5,  0.5],
            [ 0.0,  0.6,  0.0],  # Apex
        ], dtype=np.float32)

        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Base square
            (0, 4), (1, 4), (2, 4), (3, 4),  # Sides to apex
        ]


class SpatialHoloRing(SpatialObject):
    """Concentric holographic gyro-rings that idle-rotate in space."""

    def __init__(
        self,
        name: str = "GYRO-RING",
        pos: Tuple[float, float, float] = (-140.0, 0.0, 550.0),
        scale: float = 70.0,
        color: Tuple[int, int, int] = NEON_GREEN,
    ) -> None:
        super().__init__(name=name, pos=pos, scale=scale, color=color)

        num_pts = 16
        v_list = []
        e_list = []

        # Ring 1 (XY plane)
        for i in range(num_pts):
            th = 2.0 * math.pi * i / num_pts
            v_list.append([0.5 * math.cos(th), 0.5 * math.sin(th), 0.0])
            e_list.append((i, (i + 1) % num_pts))

        # Ring 2 (XZ plane)
        offset = num_pts
        for i in range(num_pts):
            th = 2.0 * math.pi * i / num_pts
            v_list.append([0.5 * math.cos(th), 0.0, 0.5 * math.sin(th)])
            e_list.append((offset + i, offset + (i + 1) % num_pts))

        self.vertices_local = np.array(v_list, dtype=np.float32)
        self.edges = e_list
