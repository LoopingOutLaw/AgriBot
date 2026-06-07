"""High-level detector: process a frame and return confirmed detections."""
from collections import deque
from dataclasses import dataclass, field
import math
import numpy as np
import cv2

from src.calibration import AxisMapping
from src.camera_math import (
    CameraIntrinsics,
    euler_to_rotation,
    pixel_to_ray,
    ray_plane_intersect,
)
from src.color_calib import ColorThresholds


@dataclass
class Detection:
    pixel_u: int
    pixel_v: int
    frame_id: int
    confidence: int
    drone_pos: np.ndarray
    drone_att: tuple[float, float, float]
    gz_x: float = 0.0
    gz_y: float = 0.0
    gz_z: float = 0.0


def color_mask(frame_bgr: np.ndarray, th: ColorThresholds | list[ColorThresholds]) -> np.ndarray:
    """Build a binary mask using HSV & LAB intersection.

    `th` may be a single ColorThresholds (intersection of HSV and LAB) or a
    list of ColorThresholds (OR-combined across each, then intersected
    with LAB per-entry).
    """
    ths = th if isinstance(th, list) else [th]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    combined = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    for t in ths:
        m_hsv = cv2.inRange(hsv, t.hsv_lower, t.hsv_upper)
        m_lab = cv2.inRange(lab, t.lab_lower, t.lab_upper)
        combined = cv2.bitwise_or(combined, cv2.bitwise_and(m_hsv, m_lab))
    return combined


def morphology_cleanup(mask: np.ndarray) -> np.ndarray:
    """Open then close with elliptical kernels tuned for 13cm crop blobs."""
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    return out


def find_largest_blob(mask: np.ndarray, min_area: int = 300) -> tuple[int, int] | None:
    """Return the (cx, cy) centroid of the largest connected blob, or None.

    Default `min_area` was raised from 100 -> 300 to suppress tiny false
    positives (drone shadow, specular highlights, edge effects) that were
    flooding the report with hundreds of FPs.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


@dataclass
class Detector:
    thresholds: ColorThresholds | list[ColorThresholds]
    intrinsics: CameraIntrinsics
    voting_window: int = 3
    confirm_threshold: int = 2
    min_centroid_jump_px: int = 50
    min_blob_area: int = 300
    plane_z: float = 0.4  # top of crop
    _window: deque = field(init=False)
    _frame_id: int = field(default=0, init=False)

    def __post_init__(self):
        # Defer deque creation until after voting_window is set
        self._window = deque(maxlen=max(10, self.voting_window))

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        drone_pos: np.ndarray,
        drone_att: tuple[float, float, float],
        axis_mapping: AxisMapping | None = None,
    ) -> Detection | None:
        self._frame_id += 1
        mask = color_mask(frame_bgr, self.thresholds)
        mask = morphology_cleanup(mask)
        centroid = find_largest_blob(mask, min_area=self.min_blob_area)
        if centroid is None:
            return None
        u, v = centroid

        # Multi-frame voting: only confirm if the centroid is stable
        self._window.append((u, v))
        if len(self._window) < self.voting_window:
            return None
        recent = list(self._window)[-self.voting_window:]
        us = [p[0] for p in recent]
        vs = [p[1] for p in recent]
        if max(us) - min(us) > self.min_centroid_jump_px:
            return None
        if max(vs) - min(vs) > self.min_centroid_jump_px:
            return None
        avg_u = int(np.mean(us))
        avg_v = int(np.mean(vs))
        confidence = sum(
            1 for p in recent if abs(p[0] - avg_u) < 20 and abs(p[1] - avg_v) < 20
        )
        if confidence < self.confirm_threshold:
            return None

        det = Detection(
            pixel_u=avg_u,
            pixel_v=avg_v,
            frame_id=self._frame_id,
            confidence=confidence,
            drone_pos=np.array(drone_pos, dtype=float),
            drone_att=tuple(drone_att),
        )
        if axis_mapping is not None:
            gx, gy, gz = self._geolocate(avg_u, avg_v, drone_pos, drone_att, axis_mapping)
            det.gz_x = gx
            det.gz_y = gy
            det.gz_z = gz
        return det

    def _geolocate(
        self,
        u: int,
        v: int,
        drone_pos: np.ndarray,
        drone_att: tuple[float, float, float],
        axis_mapping: AxisMapping,
    ) -> tuple[float, float, float]:
        """Project pixel to gz ground coordinates."""
        sign_u, sign_v = axis_mapping.sign_u, axis_mapping.sign_v
        u_use = (self.intrinsics.width - 1 - u) if sign_u == -1 else u
        v_use = (self.intrinsics.height - 1 - v) if sign_v == -1 else v
        ray_cam = pixel_to_ray(u_use, v_use, self.intrinsics)
        R_mount = euler_to_rotation(0, self.intrinsics.mount_pitch, 0)
        R_body = euler_to_rotation(*drone_att)
        R = R_body @ R_mount
        ray_world = R @ ray_cam
        cam_pos = np.array(drone_pos, dtype=float) + np.array([0, 0, -0.05])
        hit = ray_plane_intersect(cam_pos, ray_world, plane_z=self.plane_z)
        if hit is None:
            return float("nan"), float("nan"), float("nan")
        return float(hit[0]), float(hit[1]), float(hit[2])
