"""Tests for the high-level detector."""
import math
import sys
sys.path.insert(0, '/home/aditya/agribot_ws')

import numpy as np
import cv2

from src.color_calib import build_color_thresholds
from src.camera_math import CameraIntrinsics
from src.detector import (
    Detector,
    Detection,
    color_mask,
    morphology_cleanup,
    find_largest_blob,
)

HFOV = math.radians(60)
W, H = 640, 480
INTR = CameraIntrinsics(width=W, height=H, hfov=HFOV, mount_pitch=math.pi / 2)
TH = build_color_thresholds((0.6, 0.4, 0.1))


def synthetic_frame_with_yellow(cx: int, cy: int, radius: int = 30) -> np.ndarray:
    """Create a BGR frame with a green background and a yellow circle at (cx, cy)."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = (10, 100, 10)  # BGR green-ish background
    # cv2.circle takes BGR
    cv2.circle(frame, (cx, cy), radius, (25, 100, 153), -1)  # BGR for HSV (30, 213, 153)
    return frame


def test_color_mask_passes_anchor_color():
    """The exact anchor color must pass the multi-space mask."""
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[:] = (25, 100, 153)  # BGR of (0.6, 0.4, 0.1) HSV ~ (30, 213, 153)
    mask = color_mask(arr, TH)
    assert mask.sum() > 90  # most pixels should be masked


def test_color_mask_rejects_green():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[:] = (10, 100, 10)  # BGR green
    mask = color_mask(arr, TH)
    assert mask.sum() == 0


def test_morphology_removes_specks():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:12, 10:12] = 255  # 2x2 speck
    mask[40:70, 40:70] = 255  # 30x30 blob
    cleaned = morphology_cleanup(mask)
    # The 30x30 blob should survive
    assert cleaned[40:70, 40:70].sum() > 0
    # The 2x2 speck should be removed
    assert cleaned[10:12, 10:12].sum() == 0


def test_find_largest_blob_returns_centroid():
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (200, 150), 30, 255, -1)
    cx, cy = find_largest_blob(mask)
    assert abs(cx - 200) < 2
    assert abs(cy - 150) < 2


def test_find_largest_blob_returns_none_on_empty():
    mask = np.zeros((H, W), dtype=np.uint8)
    assert find_largest_blob(mask) is None


def test_detector_finds_yellow_in_synthetic_frame():
    det = Detector(thresholds=TH, intrinsics=INTR, voting_window=3)
    frame = synthetic_frame_with_yellow(W // 2 + 30, H // 2 - 20, radius=25)
    # Feed 3 frames so voting can confirm
    result = None
    for _ in range(3):
        result = det.process_frame(
            frame,
            drone_pos=np.array([0.0, 0.0, 2.0]),
            drone_att=(0.0, 0.0, 0.0),
            axis_mapping=None,  # no geolocation yet
        )
    assert result is not None
    # Centroid should be near the synthetic circle
    assert abs(result.pixel_u - (W // 2 + 30)) < 5
    assert abs(result.pixel_v - (H // 2 - 20)) < 5


def test_detector_returns_none_for_green_only_frame():
    det = Detector(thresholds=TH, intrinsics=INTR, voting_window=3)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = (10, 100, 10)  # all green
    for _ in range(3):
        result = det.process_frame(
            frame,
            drone_pos=np.array([0.0, 0.0, 2.0]),
            drone_att=(0.0, 0.0, 0.0),
            axis_mapping=None,
        )
    assert result is None
