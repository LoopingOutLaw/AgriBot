"""Tests for the disease color/type table."""
import sys
sys.path.insert(0, '/home/aditya/agribot_ws')

from src.disease_constants import DISEASE_COLORS, color_to_disease, DiseaseType


def test_all_four_disease_types_present():
    assert set(DISEASE_COLORS.keys()) == {"Healthy", "Stressed", "Rust", "Blight"}


def test_disease_colors_are_bgr_tuples():
    for name, color in DISEASE_COLORS.items():
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0.0 <= c <= 1.0 for c in color)


def test_color_to_disease_healthy_green():
    assert color_to_disease(0.1, 0.6, 0.1) == "Healthy"


def test_color_to_disease_stressed_yellow_green():
    assert color_to_disease(0.4, 0.6, 0.2) == "Stressed"


def test_color_to_disease_rust_orange():
    assert color_to_disease(0.7, 0.4, 0.1) == "Rust"


def test_color_to_disease_blight_magenta():
    assert color_to_disease(0.5, 0.1, 0.5) == "Blight"


def test_color_to_disease_picks_nearest():
    """A color close to Rust should classify as Rust."""
    assert color_to_disease(0.69, 0.41, 0.12) == "Rust"


def test_disease_type_str_enum():
    assert DiseaseType.HEALTHY.value == "Healthy"
    assert DiseaseType.BLIGHT.value == "Blight"
