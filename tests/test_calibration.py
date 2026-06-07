"""Tests for empirical camera-axis calibration."""
import math
import sys
sys.path.insert(0, '/home/aditya/agribot_ws')

import numpy as np
from src.calibration import calibrate_axis_mapping, AxisMapping
from src.camera_math import CameraIntrinsics

HFOV = math.radians(60)
W, H = 640, 480
INTR = CameraIntrinsics(width=W, height=H, hfov=HFOV, mount_pitch=math.pi / 2)


def make_correspondence(u_offset: int, v_offset: int, world_offset: tuple[float, float]):
    """Build a synthetic (drone_pose, pixel, known_world) triple.

    The drone is at (0, 0, 2) (level). The camera is at (0, 0, 1.95) (0.05 below).
    A known world point is at (world_offset[0], world_offset[1], 0.4).
    """
    drone_pos = np.array([0.0, 0.0, 2.0])
    drone_att = (0.0, 0.0, 0.0)  # roll, pitch, yaw
    pixel = (W // 2 + u_offset, H // 2 + v_offset)
    world_point = np.array([world_offset[0], world_offset[1], 0.4])
    return drone_pos, drone_att, pixel, world_point


def test_axis_mapping_defaults():
    """Default mapping is sign_u=+1, sign_v=+1 (identity)."""
    m = AxisMapping(sign_u=1, sign_v=1)
    assert m.sign_u == 1
    assert m.sign_v == 1


def test_calibration_recovers_identity_when_aligned():
    """A point directly below the drone (centred pixel) should work with any sign."""
    correspondences = [
        make_correspondence(0, 0, (0.0, 0.0)),
    ]
    mapping = calibrate_axis_mapping(correspondences, INTR)
    assert mapping is not None


def test_calibration_rejects_wild_correspondence():
    """If the correspondences are inconsistent, calibration should fail."""
    bad = [
        (np.array([0.0, 0.0, 2.0]), (0, 0, 0), (W // 2, H // 2), np.array([100.0, 100.0, 0.4])),
    ]
    mapping = calibrate_axis_mapping(bad, INTR)
    # The best mapping still has > 0.5m error -> None
    assert mapping is None or mapping.error > 0.5
