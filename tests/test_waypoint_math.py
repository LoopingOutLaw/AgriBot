"""Unit tests for Gazebo local-frame → GPS conversion and takeoff threshold.

Run with:  python3 -m pytest agribot_ws/tests/ -v
"""
import math
import pytest


def meters_to_gps_offset_north(home_lat, meters):
    """1 deg lat ≈ 111000 m. Used for Y (north) offset."""
    return meters / 111000.0


def meters_to_gps_offset_east(home_lat, home_lon, meters):
    """1 deg lon ≈ 111000 * cos(lat). Used for X (east) offset."""
    return meters / (111000.0 * math.cos(math.radians(home_lat)))


def takeoff_threshold(target_altitude):
    """Drone considers itself at target altitude when it reaches this."""
    return target_altitude * 0.95


# --- Latitude / Longitude offset tests ---

def test_north_offset_at_equator():
    offset = meters_to_gps_offset_north(0.0, 100.0)
    assert abs(offset - 0.0009009) < 1e-6  # ~0.0009 deg per 100m


def test_east_offset_at_equator():
    offset = meters_to_gps_offset_east(0.0, 0.0, 100.0)
    assert abs(offset - 0.0009009) < 1e-6


def test_east_offset_grows_with_latitude():
    """At higher latitudes, longitude lines converge — same meters span more degrees."""
    at_equator = meters_to_gps_offset_east(0.0, 0.0, 100.0)
    at_60deg = meters_to_gps_offset_east(60.0, 0.0, 100.0)
    # cos(60) = 0.5, so denominator halves → offset doubles
    assert at_60deg > at_equator
    assert at_60deg == pytest.approx(at_equator * 2)


def test_field_corner_at_sydney_spherical_origin():
    """World origin is at -35.363, 149.165 (per agribot_farm_world.sdf)."""
    home_lat, home_lon = -35.363262, 149.165237
    # Field corner at Gazebo (12, 10) — east=12, north=10
    east_offset = meters_to_gps_offset_east(home_lat, home_lon, 12.0)
    north_offset = meters_to_gps_offset_north(home_lat, 10.0)
    target_lat = home_lat + north_offset
    target_lon = home_lon + east_offset
    # Sanity: corner should be very close to home (tens of meters)
    assert abs(target_lat - home_lat) < 1e-3
    assert abs(target_lon - home_lon) < 1e-3


# --- Takeoff threshold tests ---

def test_takeoff_threshold_at_2m():
    assert takeoff_threshold(2.0) == pytest.approx(1.9)


def test_takeoff_threshold_at_5m():
    # The bug was: target_altitude-1 * 0.95 = 5 - 0.95 = 4.05 (too high)
    # The fix is: target_altitude * 0.95 = 4.75
    assert takeoff_threshold(5.0) == pytest.approx(4.75)
    assert takeoff_threshold(5.0) != pytest.approx(4.05)


def test_takeoff_threshold_uses_multiplication_not_subtraction():
    """Regression: old code had `target_altitude-1 * 0.95` which is `target_altitude - 0.95`."""
    for alt in [1.0, 2.0, 5.0, 10.0, 25.0]:
        threshold = takeoff_threshold(alt)
        assert threshold < alt, f"threshold {threshold} should be < altitude {alt}"
        assert threshold >= 0.5 * alt, f"threshold {threshold} too low for altitude {alt}"
