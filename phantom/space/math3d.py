"""PHANTOM 3D Spatial Math & Perspective Projection Engine.

Provides perspective transformation, 3D rotations, raycasting, and depth calculations
for interactive spatial computing without external heavy 3D rendering engines.
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple
import numpy as np


def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Computes composite 3D rotation matrix for angles in radians (rx, ry, rz)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # Rotation about X
    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ], dtype=np.float32)

    # Rotation about Y
    rot_y = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ], dtype=np.float32)

    # Rotation about Z
    rot_z = np.array([
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return rot_z @ rot_y @ rot_x


def project_point_3d(
    pt: np.ndarray | Tuple[float, float, float] | List[float],
    width: int,
    height: int,
    focal_length: float = 600.0,
    camera_z: float = 0.0,
) -> Optional[Tuple[int, int, float]]:
    """Projects a 3D coordinate (X, Y, Z) into 2D screen coordinates (px, py, depth).

    Returns:
        (screen_x, screen_y, depth_z) or None if point is behind camera clipping plane.
    """
    x, y, z = pt[0], pt[1], pt[2]
    rel_z = z - camera_z

    # Near-plane clipping to avoid division by zero or inversion behind camera
    if rel_z <= 20.0:
        return None

    factor = focal_length / rel_z
    screen_x = int(width / 2.0 + x * factor)
    screen_y = int(height / 2.0 - y * factor)  # Invert Y so positive Y is up
    return (screen_x, screen_y, float(rel_z))


def unproject_screen_to_ray(
    screen_x: int,
    screen_y: int,
    width: int,
    height: int,
    focal_length: float = 600.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes camera ray origin and normalized ray direction for given 2D screen coordinates."""
    ray_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dir_x = (screen_x - width / 2.0) / focal_length
    dir_y = -(screen_y - height / 2.0) / focal_length
    dir_z = 1.0

    direction = np.array([dir_x, dir_y, dir_z], dtype=np.float32)
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        direction /= norm

    return ray_origin, direction


def ray_sphere_intersect(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    sphere_center: np.ndarray,
    radius: float,
) -> Tuple[bool, float]:
    """Determines whether a ray intersects a 3D sphere.

    Returns:
        (hit: bool, distance_t: float)
    """
    oc = ray_origin - sphere_center
    b = np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - radius * radius
    discriminant = b * b - c

    if discriminant < 0:
        return False, 0.0

    t = -b - math.sqrt(discriminant)
    if t > 0:
        return True, float(t)
    t = -b + math.sqrt(discriminant)
    if t > 0:
        return True, float(t)
    return False, 0.0
