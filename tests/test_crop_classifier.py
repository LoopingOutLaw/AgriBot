"""Tests for the crop classifier."""
import sys
sys.path.insert(0, '/home/aditya/agribot_ws')

import numpy as np
import pytest

from src.crop_classifier import CropClassifier, extract_features
from src.disease_constants import DISEASE_COLORS


def test_extract_features_returns_8_d_floats():
    """A small BGR patch should produce an 8-d feature vector."""
    patch = np.full((20, 20, 3), [60, 150, 30], dtype=np.uint8)  # green
    feat = extract_features(patch)
    assert feat.shape == (8,)
    assert feat.dtype == np.float64


def test_extract_features_green_vs_red_are_different():
    """Two visually different patches should produce different feature vectors."""
    green = np.full((20, 20, 3), [30, 150, 60], dtype=np.uint8)
    red = np.full((20, 20, 3), [30, 30, 150], dtype=np.uint8)
    f_green = extract_features(green)
    f_red = extract_features(red)
    assert not np.allclose(f_green, f_red, atol=1e-3)


def test_extract_features_raises_on_empty_patch():
    with pytest.raises(ValueError):
        extract_features(np.zeros((0, 0, 3), dtype=np.uint8))


def test_classifier_trains_successfully(tmp_path):
    clf = CropClassifier(model_path=tmp_path / "c.joblib")
    clf.train()
    assert clf._trained is True
    assert clf.model is not None
    assert clf.label_encoder is not None
    assert set(clf.label_encoder.classes_) == {"Healthy", "Stressed", "Rust", "Blight"}


def test_classifier_predicts_correct_class_for_known_color():
    clf = CropClassifier()
    clf.train()
    # Green patch
    green = np.full((5, 5, 3), [30, 150, 30], dtype=np.uint8)
    p = clf.predict_proba(green)
    assert max(p, key=p.get) == "Healthy"
    # Orange (Rust) patch
    orange = np.full((5, 5, 3), [25, 100, 180], dtype=np.uint8)
    p = clf.predict_proba(orange)
    assert max(p, key=p.get) == "Rust"
    # Magenta (Blight) patch
    magenta = np.full((5, 5, 3), [127, 25, 127], dtype=np.uint8)
    p = clf.predict_proba(magenta)
    assert max(p, key=p.get) == "Blight"


def test_predict_proba_returns_4_probabilities_summing_to_1():
    clf = CropClassifier()
    clf.train()
    patch = np.full((5, 5, 3), [60, 150, 30], dtype=np.uint8)
    p = clf.predict_proba(patch)
    assert len(p) == 4
    assert abs(sum(p.values()) - 1.0) < 1e-6


def test_classifier_save_load_round_trip(tmp_path):
    path = tmp_path / "clf.joblib"
    clf = CropClassifier(model_path=path)
    clf.train()
    clf.save()

    loaded = CropClassifier.load(path)
    patch = np.full((5, 5, 3), [25, 100, 180], dtype=np.uint8)
    p_orig = clf.predict_proba(patch)
    p_loaded = loaded.predict_proba(patch)
    for k in p_orig:
        assert abs(p_orig[k] - p_loaded[k]) < 1e-6


def test_classifier_untrained_raises_on_predict():
    clf = CropClassifier()
    with pytest.raises(RuntimeError):
        clf.predict_proba(np.zeros((5, 5, 3), dtype=np.uint8))
