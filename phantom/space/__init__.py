"""PHANTOM Space – Interactive 3D Spatial Computing Engine.

Exports:
    SpatialWorkspace – 3D scene graph, object manipulation, and guided tutorial.
    SpatialObject – Base 3D interactive object class.
    SpatialCube – 3D holographic cube primitive.
    SpatialPyramid – 3D holographic pyramid primitive.
    SpatialHoloRing – 3D concentric gyro-ring primitive.
    project_point_3d – Perspective 3D to 2D projection function.
    rotation_matrix_xyz – 3D composite rotation matrix.
"""

from phantom.space.math3d import project_point_3d, rotation_matrix_xyz, unproject_screen_to_ray
from phantom.space.objects import SpatialObject, SpatialCube, SpatialPyramid, SpatialHoloRing
from phantom.space.workspace import SpatialWorkspace

__all__ = [
    "SpatialWorkspace",
    "SpatialObject",
    "SpatialCube",
    "SpatialPyramid",
    "SpatialHoloRing",
    "project_point_3d",
    "rotation_matrix_xyz",
    "unproject_screen_to_ray",
]
