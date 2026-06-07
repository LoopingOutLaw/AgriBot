"""Tests for the target-building logic in treatment_drone_controller.

We extract the pure target-building block and test it without needing
the full mission to run.
"""
import sys
sys.path.insert(0, '/home/aditya/agribot_ws')

from src.world_parser import InfectedCrop, SphericalOrigin
from src.coords import gz_to_gps


def _build_targets(detections, known_crops, origin, threshold=4):
    """Replicates the target-building block from treatment_drone_controller."""
    targets = []
    known_by_id = {c.id: c for c in (known_crops or [])}

    def _known_pos(crop_id):
        c = known_by_id.get(crop_id)
        if c is None or origin is None:
            return None
        return gz_to_gps(c.gz_x, c.gz_y, c.gz_z, origin)

    if len(detections) < threshold and known_crops and origin is not None:
        for c in known_crops:
            lat, lon, alt = gz_to_gps(c.gz_x, c.gz_y, c.gz_z, origin)
            targets.append({
                'lat': lat, 'lon': lon, 'alt': alt,
                'source': 'FALLBACK', 'crop_id': c.id,
            })
    elif detections:
        for d in detections:
            if d.get('label') == 'TRUE_POSITIVE':
                crop_id = d.get('nearest_known_id', -1)
                kp = _known_pos(crop_id)
                if kp is not None:
                    lat, lon, alt = kp
                    source = 'KNOWN_FROM_DETECT'
                else:
                    lat, lon, alt = d['lat'], d['lon'], d['alt']
                    source = 'DETECTED'
                targets.append({
                    'lat': lat, 'lon': lon, 'alt': alt,
                    'source': source, 'crop_id': crop_id,
                })
    return targets


def _make_origin():
    return SphericalOrigin(lat=-35.363261, lon=149.165230, alt=584.0)


def _make_known():
    return [
        InfectedCrop(id=8, gz_x=0.0, gz_y=0.9, gz_z=0.2),
        InfectedCrop(id=119, gz_x=3.0, gz_y=4.4, gz_z=0.2),
    ]


def test_known_from_detect_uses_known_position_not_detected_gps():
    """When a detection is matched to a known crop, treatment must use the
    known world-file position (not the noisy camera-computed GPS)."""
    origin = _make_origin()
    known = _make_known()
    known_lat, known_lon, known_alt = gz_to_gps(0.0, 0.9, 0.2, origin)

    # Pad detections to >= threshold so the FALLBACK branch is skipped
    detections = [
        {'label': 'TRUE_POSITIVE', 'lat': known_lat + 0.0001, 'lon': known_lon + 0.0001,
         'alt': known_alt, 'nearest_known_id': 8},
    ] + [
        {'label': 'FALSE_POSITIVE', 'lat': 0, 'lon': 0, 'alt': 0, 'nearest_known_id': -1}
        for _ in range(4)
    ]
    targets = _build_targets(detections, known, origin)
    true_targets = [t for t in targets if t['source'] == 'KNOWN_FROM_DETECT']
    assert len(true_targets) == 1
    t = true_targets[0]
    assert t['crop_id'] == 8
    assert abs(t['lat'] - known_lat) < 1e-7
    assert abs(t['lon'] - known_lon) < 1e-7
    assert abs(t['alt'] - known_alt) < 1e-7


def test_fallback_uses_known_position_when_few_detections():
    origin = _make_origin()
    known = _make_known()
    detections = [{'label': 'TRUE_POSITIVE', 'lat': 0, 'lon': 0, 'alt': 0,
                   'nearest_known_id': 8}]
    targets = _build_targets(detections, known, origin, threshold=4)
    assert len(targets) == 2
    for t in targets:
        assert t['source'] == 'FALLBACK'
        assert t['crop_id'] in {8, 119}


def test_detect_without_known_match_uses_detected_position():
    """If the detection has no matching known crop, fall back to the noisy
    detected position (this is rare; the rule is to mark the source)."""
    origin = _make_origin()
    known = _make_known()
    detections = [
        {'label': 'TRUE_POSITIVE', 'lat': 0.0001, 'lon': 0.0001, 'alt': 584.5,
         'nearest_known_id': -1},  # no known match
    ] + [
        {'label': 'FALSE_POSITIVE', 'lat': 0, 'lon': 0, 'alt': 0, 'nearest_known_id': -1}
        for _ in range(4)
    ]
    targets = _build_targets(detections, known, origin)
    detected_targets = [t for t in targets if t['source'] == 'DETECTED']
    assert len(detected_targets) == 1
    assert detected_targets[0]['source'] == 'DETECTED'


def test_no_detections_no_known_no_targets():
    origin = _make_origin()
    targets = _build_targets([], [], origin)
    assert targets == []
