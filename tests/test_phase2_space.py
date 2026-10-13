"""Automated Unit & Integration Test Suite for PHANTOM Space (Phase 2)."""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phantom.space.math3d import (
    project_point_3d,
    rotation_matrix_xyz,
    unproject_screen_to_ray,
    ray_sphere_intersect,
)
from phantom.space.objects import SpatialObject, SpatialCube, SpatialPyramid, SpatialHoloRing
from phantom.space.workspace import SpatialWorkspace


class TestPhantomSpace(unittest.TestCase):
    """Verifies all components of Phase 2 - PHANTOM Space."""

    def setUp(self):
        self.width = 640
        self.height = 480
        self.dummy_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def test_01_3d_projection(self):
        """Test perspective projection of 3D points to screen coords."""
        # Point right in center in front of camera
        pt = (0.0, 0.0, 600.0)
        proj = project_point_3d(pt, self.width, self.height, focal_length=600.0)
        self.assertIsNotNone(proj)
        self.assertEqual((proj[0], proj[1]), (self.width // 2, self.height // 2))

        # Behind-camera clipping check
        behind_pt = (0.0, 0.0, 10.0)
        self.assertIsNone(project_point_3d(behind_pt, self.width, self.height, focal_length=600.0))

    def test_02_rotation_matrix(self):
        """Test 3D composite rotation matrix properties."""
        r_mat = rotation_matrix_xyz(0.0, 0.0, 0.0)
        np.testing.assert_allclose(r_mat, np.eye(3, dtype=np.float32), atol=1e-5)

        # Orthogonality check for arbitrary angles
        r_mat_rand = rotation_matrix_xyz(0.3, 0.7, -0.4)
        np.testing.assert_allclose(r_mat_rand @ r_mat_rand.T, np.eye(3, dtype=np.float32), atol=1e-5)

    def test_03_raycasting(self):
        """Test ray unprojection and sphere intersection."""
        ray_origin, ray_dir = unproject_screen_to_ray(self.width // 2, self.height // 2, self.width, self.height)
        self.assertAlmostEqual(np.linalg.norm(ray_dir), 1.0, places=4)

        sphere_center = np.array([0.0, 0.0, 500.0], dtype=np.float32)
        hit, dist = ray_sphere_intersect(ray_origin, ray_dir, sphere_center, radius=50.0)
        self.assertTrue(hit)
        self.assertGreater(dist, 0.0)

    def test_04_spatial_primitives(self):
        """Test Cube, Pyramid, and GyroRing vertex definitions and rendering."""
        cube = SpatialCube(pos=(0.0, 0.0, 500.0), scale=50.0)
        self.assertEqual(len(cube.vertices_local), 8)
        self.assertEqual(len(cube.edges), 12)

        pyramid = SpatialPyramid(pos=(50.0, 0.0, 500.0), scale=50.0)
        self.assertEqual(len(pyramid.vertices_local), 5)

        ring = SpatialHoloRing(pos=(-50.0, 0.0, 500.0), scale=50.0)
        self.assertGreater(len(ring.vertices_local), 10)

        # Render test
        frame = self.dummy_frame.copy()
        cube.render(frame, self.width, self.height)
        pyramid.render(frame, self.width, self.height)
        ring.render(frame, self.width, self.height)
        self.assertTrue(np.any(frame != 0))

    def test_05_single_hand_manipulation(self):
        """Test single-hand pinch grab and 3D translation."""
        cube = SpatialCube(pos=(0.0, 0.0, 500.0), scale=50.0)
        initial_pos = np.copy(cube.pos)

        # Start grab at screen center
        cube.start_single_grab(hand_idx=0, screen_pos=(320, 240))
        self.assertTrue(cube.is_grabbed)

        # Move screen coordinate to the right
        cube.update_single_grab(
            hand_idx=0,
            curr_screen_pos=(370, 240),
            palm_scale_ratio=1.0,
            width=self.width,
            height=self.height,
        )
        self.assertGreater(cube.pos[0], initial_pos[0])

        # Release grab
        cube.release_grab(hand_idx=0)
        self.assertFalse(cube.is_grabbed)

    def test_06_dual_hand_manipulation(self):
        """Test two-hand dual pinch scaling and rotation."""
        cube = SpatialCube(pos=(0.0, 0.0, 500.0), scale=50.0)
        initial_scale = cube.scale

        cube.start_dual_grab(h0_pos=(280, 240), h1_pos=(360, 240))
        self.assertTrue(cube.is_grabbed)
        self.assertEqual(len(cube.grabbed_by), 2)

        # Pull hands apart to scale up
        cube.update_dual_grab(
            h0_pos=(240, 240),
            h1_pos=(400, 240),
            width=self.width,
            height=self.height,
        )
        self.assertGreater(cube.scale, initial_scale)

        cube.release_grab()
        self.assertFalse(cube.is_grabbed)

    def test_07_workspace_scene_graph(self):
        """Test full 3D workspace, interaction loop, and rendering."""
        workspace = SpatialWorkspace(width=self.width, height=self.height)
        self.assertEqual(len(workspace.objects), 3)

        # Update with empty hands
        workspace.update_and_interact([], dt=0.033)

        # Update with a synthetic hand hovering
        syn_hand = [{
            "hand_index": 0,
            "handedness": "Right",
            "index_tip": (self.width // 2, self.height // 2),
            "pinch": False,
            "palm_scale": 60.0,
            "index_up": True,
            "palm": False,
        }]
        workspace.update_and_interact(syn_hand, dt=0.033)

        # Render workspace
        frame = self.dummy_frame.copy()
        workspace.render(frame)
        self.assertTrue(np.any(frame != 0))


if __name__ == "__main__":
    unittest.main()
