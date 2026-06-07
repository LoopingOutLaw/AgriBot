"""Tests for gz <-> GPS coordinate conversion."""
import math
import sys
sys.path.insert(0, '/home/aditya/agribot_ws')

from src.world_parser import SphericalOrigin
from src.coords import gz_to_gps, gps_to_gz

ORIGIN = SphericalOrigin(lat=-35.363261, lon=149.165230, alt=584.0)
M_PER_DEG_LAT = (math.pi / 180) * 6_371_000
M_PER_DEG_LON = M_PER_DEG_LAT * math.cos(math.radians(ORIGIN.lat))


def test_origin_maps_to_zero():
    lat, lon, alt = gz_to_gps(0.0, 0.0, 0.0, ORIGIN)
    assert abs(lat - ORIGIN.lat) < 1e-9
    assert abs(lon - ORIGIN.lon) < 1e-9
    assert abs(alt - ORIGIN.alt) < 1e-9


def test_one_meter_north():
    """1 m north (+Y in gz) should give a small positive dlat."""
    lat, lon, _ = gz_to_gps(0.0, 1.0, 0.0, ORIGIN)
    expected_dlat = 1.0 / M_PER_DEG_LAT
    assert abs((lat - ORIGIN.lat) - expected_dlat) < 1e-12


def test_one_meter_east():
    """1 m east (+X in gz) should give a small positive dlon."""
    lat, lon, _ = gz_to_gps(1.0, 0.0, 0.0, ORIGIN)
    expected_dlon = 1.0 / M_PER_DEG_LON
    assert abs((lon - ORIGIN.lon) - expected_dlon) < 1e-12


def test_one_meter_up():
    """1 m up (+Z in gz) should give a 1 m altitude offset."""
    _, _, alt = gz_to_gps(0.0, 0.0, 1.0, ORIGIN)
    assert abs(alt - (ORIGIN.alt + 1.0)) < 1e-9


def test_round_trip_zero():
    """gz→gps→gz should be lossless at origin."""
    x0, y0, z0 = 0.0, 0.0, 0.0
    lat, lon, alt = gz_to_gps(x0, y0, z0, ORIGIN)
    x, y, z = gps_to_gz(lat, lon, alt, ORIGIN)
    assert abs(x - x0) < 1e-9
    assert abs(y - y0) < 1e-9
    assert abs(z - z0) < 1e-9


def test_round_trip_typical_crop():
    """A typical crop position should round-trip with < 1 mm error."""
    x0, y0, z0 = 1.2, 4.0, 0.4  # crop_119
    lat, lon, alt = gz_to_gps(x0, y0, z0, ORIGIN)
    x, y, z = gps_to_gz(lat, lon, alt, ORIGIN)
    assert abs(x - x0) < 1e-3, f"x error: {abs(x - x0)} m"
    assert abs(y - y0) < 1e-3, f"y error: {abs(y - y0)} m"
    assert abs(z - z0) < 1e-3, f"z error: {abs(z - z0)} m"


def test_round_trip_far_corner():
    """The far corner of the field should also round-trip tightly."""
    x0, y0, z0 = -1.4, 5.4, 0.4
    lat, lon, alt = gz_to_gps(x0, y0, z0, ORIGIN)
    x, y, z = gps_to_gz(lat, lon, alt, ORIGIN)
    assert abs(x - x0) < 1e-3
    assert abs(y - y0) < 1e-3
