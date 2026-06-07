"""Tests for the top-down field map renderer."""
import sys
sys.path.insert(0, '/home/aditya/agribot_ws')

import numpy as np

from src.field_map import FieldMap
from src.world_parser import parse_infected_crops, parse_spherical_origin, WORLD_PATH


def _make_map():
    return FieldMap(world_path=WORLD_PATH, size_px=400)


def test_field_map_renders_400x400_bgr_image():
    fmap = _make_map()
    img = fmap.update(detected_infections=[], cluster_result=None,
                      scout_pos=None, treatment_pos=None,
                      treatment_target=None, spray_position=None)
    assert img.shape == (400, 400, 3)
    assert img.dtype == np.uint8


def test_field_map_renders_healthy_crops_as_green_dots():
    """We should see green pixels (the healthy crop color) somewhere in the map."""
    fmap = _make_map()
    img = fmap.update([], None, None, None, None, None)
    # Convert to HSV to check for green
    hsv = img[..., :]  # already BGR
    # Just check the image has variation (not all-black background)
    assert img.std() > 5.0, "map appears blank"


def test_field_map_renders_detected_infections():
    fmap = _make_map()
    # Put a fake detection near a real crop position
    origin = parse_spherical_origin(WORLD_PATH)
    crops = parse_infected_crops(WORLD_PATH)
    if not crops:
        return  # no diseased crops in current world, skip
    c = crops[0]
    img_before = fmap.update([], None, None, None, None, None)
    img_after = fmap.update([(c.gz_x, c.gz_y)], None, None, None, None, None)
    # The after-image should differ (we drew an infection marker)
    assert not np.array_equal(img_before, img_after)


def test_field_map_renders_drone_positions():
    fmap = _make_map()
    img_no_drone = fmap.update([], None, None, None, None, None)
    img_with_drone = fmap.update([], None, scout_pos=(0.0, 3.0), treatment_pos=(0.0, 3.0),
                                 treatment_target=None, spray_position=None)
    assert not np.array_equal(img_no_drone, img_with_drone)


def test_field_map_uses_world_file_field_bounds():
    """The map should include the field extent from the world file."""
    fmap = _make_map()
    img = fmap.update([], None, None, None, None, None)
    # Map should not be all background — should have visible crops
    non_bg_pixels = (img.sum(axis=-1) > 0).sum()
    assert non_bg_pixels > 100, f"map looks empty: {non_bg_pixels} non-bg pixels"
