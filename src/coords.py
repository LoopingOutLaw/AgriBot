"""Gazebo <-> GPS coordinate conversion (local tangent plane)."""
import math

from src.world_parser import SphericalOrigin

R_EARTH = 6_371_000.0  # m
DEG = math.pi / 180.0


def _meters_per_degree(origin: SphericalOrigin) -> tuple[float, float]:
    m_per_deg_lat = DEG * R_EARTH
    m_per_deg_lon = m_per_deg_lat * math.cos(origin.lat * DEG)
    return m_per_deg_lat, m_per_deg_lon


def gz_to_gps(
    x: float, y: float, z: float, origin: SphericalOrigin
) -> tuple[float, float, float]:
    """Convert gz (east, north, up) to (lat, lon, alt).

    Convention: gz +X = east, +Y = north, +Z = up (ENU).
    """
    m_per_deg_lat, m_per_deg_lon = _meters_per_degree(origin)
    lat = origin.lat + (y / m_per_deg_lat)
    lon = origin.lon + (x / m_per_deg_lon)
    alt = origin.alt + z
    return lat, lon, alt


def gps_to_gz(
    lat: float, lon: float, alt: float, origin: SphericalOrigin
) -> tuple[float, float, float]:
    """Convert (lat, lon, alt) to gz (east, north, up)."""
    m_per_deg_lat, m_per_deg_lon = _meters_per_degree(origin)
    x = (lon - origin.lon) * m_per_deg_lon
    y = (lat - origin.lat) * m_per_deg_lat
    z = alt - origin.alt
    return x, y, z
