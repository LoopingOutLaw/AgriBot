"""Top-down field map renderer for the GUI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.world_parser import (
    parse_infected_crops, parse_spherical_origin, WORLD_PATH, InfectedCrop,
)
from src.infection_clustering import ClusterResult
from src.disease_constants import DISEASE_COLORS


@dataclass
class _CropPos:
    gz_x: float
    gz_y: float
    disease: str


class FieldMap:
    """Renders a top-down BGR image of the field with crops, infections,
    cluster centers, and drone positions."""

    BG_COLOR = (40, 30, 20)  # dark olive in BGR
    PAD_M = 0.5              # extra meters around the field

    def __init__(self, world_path: Path = WORLD_PATH, size_px: int = 400):
        self.size_px = size_px
        origin = parse_spherical_origin(world_path)
        all_crops = parse_infected_crops(world_path)

        # Field bounds: cover all diseased crops + some padding
        if all_crops:
            xs = [c.gz_x for c in all_crops]
            ys = [c.gz_y for c in all_crops]
            self.x_min = min(xs) - self.PAD_M
            self.x_max = max(xs) + self.PAD_M
            self.y_min = min(ys) - self.PAD_M
            self.y_max = max(ys) + self.PAD_M
        else:
            self.x_min, self.x_max = -3.0, 3.0
            self.y_min, self.y_max = 0.0, 8.0

        # Build full crop list: 221 healthy + 11 diseased
        # We use the 8x8 = 64-grid approximation: 15x15 = 225 crops, healthy at non-diseased positions
        diseased_ids = {c.id for c in all_crops}
        self._crops: list[_CropPos] = []
        for c in all_crops:
            self._crops.append(_CropPos(c.gz_x, c.gz_y, c.disease_type))
        # Add healthy crops at synthetic positions
        for idx in range(1, 226):
            if idx in diseased_ids:
                continue
            col = (idx - 1) % 15
            row = (idx - 1) // 15
            x = -3.5 + col * 0.5
            y = 0.9 + row * 0.5
            self._crops.append(_CropPos(x, y, "Healthy"))

    def _gz_to_px(self, gz_x: float, gz_y: float) -> tuple[int, int]:
        u = (gz_x - self.x_min) / (self.x_max - self.x_min)
        v = (gz_y - self.y_min) / (self.y_max - self.y_min)
        # gz y is north, screen y is down — flip
        px = int(u * (self.size_px - 1))
        py = int((1.0 - v) * (self.size_px - 1))
        return px, py

    def _bgr_from_rgb(self, rgb: tuple[float, float, float]) -> tuple[int, int, int]:
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

    def update(
        self,
        detected_infections: list[tuple[float, float]],
        cluster_result: ClusterResult | None,
        scout_pos: tuple[float, float] | None,
        treatment_pos: tuple[float, float] | None,
        treatment_target: tuple[float, float] | None,
        spray_position: tuple[float, float] | None,
    ) -> np.ndarray:
        """Render and return a BGR image of the field."""
        img = np.full((self.size_px, self.size_px, 3), self.BG_COLOR, dtype=np.uint8)

        # Field boundary
        tl = self._gz_to_px(self.x_min, self.y_max)
        br = self._gz_to_px(self.x_max, self.y_min)
        cv2.rectangle(img, tl, br, (60, 50, 40), 1)

        # All crops as small colored dots
        for crop in self._crops:
            px, py = self._gz_to_px(crop.gz_x, crop.gz_y)
            color = self._bgr_from_rgb(DISEASE_COLORS[crop.disease])
            cv2.circle(img, (px, py), 4, color, -1)

        # Detected infections: outline + label
        for gx, gy in detected_infections:
            px, py = self._gz_to_px(gx, gy)
            cv2.circle(img, (px, py), 8, (0, 0, 255), 2)

        # Cluster centers: large semi-transparent circles
        if cluster_result is not None:
            overlay = img.copy()
            for i, (cx, cy) in enumerate(cluster_result.centers):
                px, py = self._gz_to_px(cx, cy)
                cv2.circle(overlay, (px, py), 20, (255, 200, 0), -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            for i, (cx, cy) in enumerate(cluster_result.centers):
                px, py = self._gz_to_px(cx, cy)
                cv2.circle(img, (px, py), 20, (255, 200, 0), 2)
                cv2.putText(img, f"C{i}", (px - 8, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Drone positions
        if scout_pos is not None:
            px, py = self._gz_to_px(*scout_pos)
            cv2.drawMarker(img, (px, py), (0, 255, 0),
                           markerType=cv2.MARKER_TRIANGLE_UP, markerSize=12, thickness=2)
            cv2.putText(img, "S", (px + 8, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if treatment_pos is not None:
            px, py = self._gz_to_px(*treatment_pos)
            cv2.drawMarker(img, (px, py), (255, 100, 0),
                           markerType=cv2.MARKER_TRIANGLE_UP, markerSize=12, thickness=2)
            cv2.putText(img, "T", (px + 8, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

        # Treatment target line
        if treatment_pos is not None and treatment_target is not None:
            p1 = self._gz_to_px(*treatment_pos)
            p2 = self._gz_to_px(*treatment_target)
            cv2.line(img, p1, p2, (255, 255, 0), 1, cv2.LINE_AA)

        # Spray effect
        if spray_position is not None:
            px, py = self._gz_to_px(*spray_position)
            cv2.circle(img, (px, py), 25, (180, 255, 180), 2)
            cv2.circle(img, (px, py), 15, (200, 255, 200), 2)

        # Title
        cv2.putText(img, "FIELD MAP (top-down)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "S=Scout T=Treatment", (10, self.size_px - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return img
