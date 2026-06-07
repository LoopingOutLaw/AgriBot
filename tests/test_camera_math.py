"""Tests for camera math: pinhole intrinsics, ray construction."""
import math
import sys
sys.path.insert(0, '/home/aditya/agribot_ws')

import numpy as np
from src.camera_math import (
    derive_vfov,
    pixel_to_ray,
    ray_plane_intersect,
    euler_to_rotation,
    CameraIntrinsics,
)

HFOV = math.radians(60)
W, H = 640, 480
INTR = CameraIntrinsics(width=W, height=H, hfov=HFOV, mount_pitch=math.pi / 2)


def test_vfov_from_hfov_and_aspect():
    vfov = derive_vfov(HFOV, W, H)
    expected = 2 * math.atan((H / W) * math.tan(HFOV / 2))
    assert abs(vfov - expected) < 1e-9


def test_vfov_45deg_for_4_3():
    """For 60 deg HFOV and 4:3 aspect, VFOV is ~46.8 deg (not 45)."""
    vfov_deg = math.degrees(derive_vfov(HFOV, 640, 480))
    assert 46 < vfov_deg < 47


def test_pixel_to_ray_center():
    """Centre pixel -> ray straight forward (camera's +X axis)."""
    ray = pixel_to_ray(W // 2, H // 2, INTR)
    assert abs(ray[1]) < 1e-9
    assert abs(ray[2]) < 1e-9
    assert abs(ray[0] - 1.0) < 1e-9


def test_pixel_to_ray_top_right():
    """Top-right pixel: y > 0 (right of centre), z < 0 (image top, where v is small)."""
    ray = pixel_to_ray(W - 1, 0, INTR)
    assert ray[0] > 0   # forward (always positive for any pixel)
    assert ray[1] > 0   # right of centre in image -> +Y in camera frame
    assert ray[2] < 0   # top of image (v=0) gives z < 0 in camera frame
    # Magnitude is 1 (unit vector)
    assert abs(np.linalg.norm(ray) - 1.0) < 1e-9


def test_pixel_to_ray_unit_vectors():
    """All rays should be unit vectors."""
    for u, v in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1), (W // 2, H // 2)]:
        ray = pixel_to_ray(u, v, INTR)
        assert abs(np.linalg.norm(ray) - 1.0) < 1e-9


def test_ray_plane_intersect_vertical():
    """A ray pointing straight down (z=-1) from (0, 0, 2) hits z=0 at (0, 0, 0)."""
    ray = np.array([0.0, 0.0, -1.0])
    cam = np.array([0.0, 0.0, 2.0])
    point = ray_plane_intersect(cam, ray, plane_z=0.0)
    assert np.allclose(point, [0.0, 0.0, 0.0], atol=1e-9)


def test_ray_plane_intersect_angled():
    """A 45 deg ray from (0, 0, 2) hits z=0 at (2, 0, 0)."""
    ray = np.array([1.0, 0.0, -1.0])  # 45 deg
    cam = np.array([0.0, 0.0, 2.0])
    point = ray_plane_intersect(cam, ray, plane_z=0.0)
    assert abs(point[0] - 2.0) < 1e-9
    assert abs(point[1] - 0.0) < 1e-9
    assert abs(point[2] - 0.0) < 1e-9


def test_ray_plane_intersect_rejects_upward_ray():
    """A ray pointing up should return None (no ground hit)."""
    ray = np.array([0.0, 0.0, 1.0])
    cam = np.array([0.0, 0.0, 2.0])
    point = ray_plane_intersect(cam, ray, plane_z=0.0)
    assert point is None


def test_euler_to_rotation_identity():
    R = euler_to_rotation(0, 0, 0)
    assert np.allclose(R, np.eye(3), atol=1e-9)


def test_euler_to_rotation_pitch_90():
    """Pitching 90 deg about Y should rotate +X to -Z and +Z to +X."""
    R = euler_to_rotation(0, math.pi / 2, 0)
    x_rot = R @ np.array([1.0, 0.0, 0.0])
    z_rot = R @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(x_rot, [0, 0, -1], atol=1e-9)  # +X -> -Z
    assert np.allclose(z_rot, [1, 0, 0], atol=1e-9)   # +Z -> +X
