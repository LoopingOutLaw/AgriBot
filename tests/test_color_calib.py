"""Tests for color calibration from world file."""
import sys
sys.path.insert(0, '/home/aditya/agribot_ws')

import numpy as np
from src.color_calib import (
    rgb_to_hsv,
    rgb_to_lab,
    rgb_to_hsl,
    build_color_thresholds,
    ColorThresholds,
)

INFECTED_RGB = (0.6, 0.4, 0.1)


def test_rgb_to_hsv_yellow_brown():
    """RGB (0.6, 0.4, 0.1) is a yellow-brown. In OpenCV H=[0,180], hue ~18."""
    h, s, v = rgb_to_hsv(*INFECTED_RGB)
    # OpenCV uses H in [0, 180]; the actual hue is ~18 (yellow-orange)
    assert 15 <= h <= 21, f"Hue {h} not in [15, 21]"
    assert s > 200, f"Saturation {s} should be high"
    assert v > 140, f"Value {v} should be moderate-high"


def test_rgb_to_lab_yellow():
    """Yellow-brown has high b*, moderate L*."""
    L, a, b = rgb_to_lab(*INFECTED_RGB)
    assert b > 40, f"b* {b} should be > 40 for yellow"
    assert L > 30, f"L* {L} should be > 30"


def test_rgb_to_hsl_yellow_brown():
    h, s, l = rgb_to_hsl(*INFECTED_RGB)
    assert 15 <= h <= 21
    assert l > 30


def test_thresholds_cover_anchor_color():
    """The thresholds must accept the anchor color exactly."""
    th = build_color_thresholds(INFECTED_RGB)
    assert isinstance(th, ColorThresholds)
    # The anchor should be inside the HSV range
    h, s, v = rgb_to_hsv(*INFECTED_RGB)
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    import cv2
    mask = cv2.inRange(hsv, th.hsv_lower, th.hsv_upper)
    assert mask[0, 0] == 255, "Anchor color must pass HSV mask"


def test_thresholds_have_tolerance():
    """A slightly different yellow should also pass."""
    th = build_color_thresholds(INFECTED_RGB)
    h, s, v = rgb_to_hsv(0.55, 0.4, 0.15)
    import cv2
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    mask = cv2.inRange(hsv, th.hsv_lower, th.hsv_upper)
    assert mask[0, 0] == 255, "Slightly lighter yellow should pass"


def test_thresholds_reject_green():
    """Healthy green (0.1, 0.6, 0.1) must be rejected."""
    th = build_color_thresholds(INFECTED_RGB)
    h, s, v = rgb_to_hsv(0.1, 0.6, 0.1)
    import cv2
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    mask = cv2.inRange(hsv, th.hsv_lower, th.hsv_upper)
    assert mask[0, 0] == 0, "Green must not pass HSV mask"
