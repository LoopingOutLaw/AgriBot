"""Crop health classifier: trains a small scikit-learn model on synthetic
color patches and predicts disease type + confidence from a BGR patch."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.disease_constants import DISEASE_COLORS

logger = logging.getLogger(__name__)


def extract_features(patch_bgr: np.ndarray) -> np.ndarray:
    """Compute an 8-d feature vector from a BGR patch.

    Features: HSV mean (3) + LAB mean (3) + raw R/G channels normalized to [0, 1] (2).
    """
    if patch_bgr.size == 0 or patch_bgr.shape[0] == 0 or patch_bgr.shape[1] == 0:
        raise ValueError("patch is empty")

    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)

    hsv_mean = hsv.reshape(-1, 3).mean(axis=0) / np.array([180.0, 255.0, 255.0])
    lab_mean = lab.reshape(-1, 3).mean(axis=0) / np.array([255.0, 255.0, 255.0])
    # BGR -> take R and G channels, normalize to [0, 1]
    bgr_mean = patch_bgr.reshape(-1, 3).mean(axis=0) / 255.0
    rg = np.array([bgr_mean[2], bgr_mean[1]])  # R, G

    return np.concatenate([hsv_mean, lab_mean, rg]).astype(np.float64)


class CropClassifier:
    """Trains and predicts crop health from color patches."""

    MODEL_FILENAME = "crop_classifier.joblib"

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path
        self.model: LogisticRegression | None = None
        self.label_encoder: LabelEncoder | None = None
        self._trained = False

    def train(self, samples_per_class: int = 50, noise_std: float = 0.04,
              random_state: int = 42) -> None:
        """Train on synthetic data generated from the DISEASE_COLORS table."""
        rng = np.random.default_rng(random_state)
        X_list: list[np.ndarray] = []
        y_list: list[str] = []

        for disease, (r, g, b) in DISEASE_COLORS.items():
            for _ in range(samples_per_class):
                rj = r + rng.normal(0, noise_std)
                gj = g + rng.normal(0, noise_std)
                bj = b + rng.normal(0, noise_std)
                rgb = np.clip([rj, gj, bj], 0, 1)
                bgr_pix = np.array([
                    [int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)]
                ], dtype=np.uint8)
                # Make a 5x5 patch
                patch = np.broadcast_to(bgr_pix, (5, 5, 3)).copy()
                X_list.append(extract_features(patch))
                y_list.append(disease)

        X = np.stack(X_list)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_list)

        self.model = LogisticRegression(max_iter=500, multi_class="multinomial")
        self.model.fit(X, y)
        self._trained = True
        logger.info(
            f"Trained crop classifier on {len(X)} synthetic samples "
            f"({len(self.label_encoder.classes_)} classes)"
        )

    def predict_proba(self, patch_bgr: np.ndarray) -> dict[str, float]:
        """Return {class_name: probability} for the given BGR patch."""
        if not self._trained:
            raise RuntimeError("classifier not trained; call train() first or load()")
        feat = extract_features(patch_bgr).reshape(1, -1)
        probs = self.model.predict_proba(feat)[0]
        return {
            cls: float(p)
            for cls, p in zip(self.label_encoder.classes_, probs)
        }

    def save(self, path: Path | None = None) -> None:
        target = path or self.model_path
        if target is None:
            raise ValueError("no model_path provided")
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "label_encoder": self.label_encoder}, target)
        logger.info(f"Saved classifier to {target}")

    @classmethod
    def load(cls, path: Path) -> "CropClassifier":
        data = joblib.load(path)
        inst = cls(model_path=path)
        inst.model = data["model"]
        inst.label_encoder = data["label_encoder"]
        inst._trained = True
        return inst
