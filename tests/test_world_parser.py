"""Tests for world file parser."""
import sys
sys.path.insert(0, '/home/aditya/agribot_ws')

from src.world_parser import parse_infected_crops, parse_spherical_origin, WORLD_PATH
from src.disease_constants import color_to_disease

EXPECTED_INFECTED_IDS = {1, 8, 30, 50, 80, 100, 119, 157, 158, 175, 224}
EXPECTED_INFECTED_POSES = {
    1: (-3.500, 0.900),
    8: (0.000, 0.900),
    30: (3.500, 1.400),
    50: (-1.500, 2.400),
    80: (-1.500, 3.400),
    100: (1.000, 3.900),
    119: (3.000, 4.400),
    157: (-0.500, 5.900),
    158: (0.000, 5.900),
    175: (1.000, 6.400),
    224: (3.000, 7.900),
}


def test_world_file_exists():
    assert WORLD_PATH.exists(), f"World file not found at {WORLD_PATH}"


def test_parse_finds_exactly_11_diseased_crops():
    crops = parse_infected_crops(WORLD_PATH)
    assert len(crops) == 11, f"Expected 11 diseased, got {len(crops)}"


def test_parse_returns_expected_ids():
    crops = parse_infected_crops(WORLD_PATH)
    ids = {c.id for c in crops}
    assert ids == EXPECTED_INFECTED_IDS


def test_parse_returns_correct_positions():
    crops = parse_infected_crops(WORLD_PATH)
    by_id = {c.id: c for c in crops}
    for cid, (ex, ey) in EXPECTED_INFECTED_POSES.items():
        c = by_id[cid]
        assert abs(c.gz_x - ex) < 1e-6, f"crop_{cid} x: {c.gz_x} != {ex}"
        assert abs(c.gz_y - ey) < 1e-6, f"crop_{cid} y: {c.gz_y} != {ey}"


def test_parse_excludes_healthy_crops():
    """The parser must not return any of the green healthy crops (sample)."""
    crops = parse_infected_crops(WORLD_PATH)
    by_id = {c.id for c in crops}
    for healthy_id in {2, 3, 4, 5, 6, 7, 9, 10, 11, 12}:
        assert healthy_id not in by_id, f"healthy crop_{healthy_id} should be excluded"


def test_parse_spherical_origin_lat():
    origin = parse_spherical_origin(WORLD_PATH)
    assert abs(origin.lat - (-35.363261)) < 1e-6


def test_parse_spherical_origin_lon():
    origin = parse_spherical_origin(WORLD_PATH)
    assert abs(origin.lon - 149.165230) < 1e-6


def test_parse_spherical_origin_alt():
    origin = parse_spherical_origin(WORLD_PATH)
    assert abs(origin.alt - 584.0) < 1e-3


def test_parsed_crops_have_disease_type():
    crops = parse_infected_crops(WORLD_PATH)
    valid = {"Healthy", "Stressed", "Rust", "Blight"}
    for c in crops:
        assert c.disease_type in valid, f"crop {c.id} has invalid disease_type {c.disease_type}"


def test_disease_type_matches_world_color():
    """Each parsed crop's disease_type should match the color of its <material> tag."""
    import re
    text = open(WORLD_PATH).read()
    crops = parse_infected_crops(WORLD_PATH)
    for c in crops:
        m = re.search(
            rf'<model name="crop_{c.id}">.*?<ambient>([\d.]+) ([\d.]+) ([\d.]+)',
            text, re.DOTALL,
        )
        assert m is not None, f"crop {c.id} has no <ambient> tag"
        r, g, b = float(m.group(1)), float(m.group(2)), float(m.group(3))
        assert c.disease_type == color_to_disease(r, g, b)


def test_parse_includes_eleven_diseased_crops():
    """11 crops should be classified as diseased (5 Stressed + 6 Rust, 0 Blight).

    Note: Blight (magenta) was dropped from the world because the rendered
    color wasn't being detected reliably. The 4 magenta crops were
    redistributed to Stressed/Rust to keep 11 diseased total.
    """
    crops = parse_infected_crops(WORLD_PATH)
    assert len(crops) == 11
    counts = {}
    for c in crops:
        counts[c.disease_type] = counts.get(c.disease_type, 0) + 1
    assert counts == {"Stressed": 5, "Rust": 6}
