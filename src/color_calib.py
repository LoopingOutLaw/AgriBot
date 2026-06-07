"""Color calibration: convert anchor RGB to HSV/HSL/LAB and build thresholds."""
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass(frozen=True)
class ColorThresholds:
    hsv_lower: np.ndarray  # uint8, shape (3,)
    hsv_upper: np.ndarray
    hsl_lower: np.ndarray
    hsl_upper: np.ndarray
    lab_lower: np.ndarray
    lab_upper: np.ndarray


def rgb_to_hsv(r: float, g: float, b: float) -> tuple[int, int, int]:
    """RGB in [0, 1] -> HSV in OpenCV uint8 ranges (H: 0-180, S: 0-255, V: 0-255)."""
    arr = np.array([[[b * 255, g * 255, r * 255]]], dtype=np.uint8)  # cv2 BGR
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    h, s, v = int(hsv[0, 0, 0]), int(hsv[0, 0, 1]), int(hsv[0, 0, 2])
    return h, s, v


def rgb_to_hsl(r: float, g: float, b: float) -> tuple[int, int, int]:
    """RGB in [0, 1] -> HSL (H: 0-180, S: 0-255, L: 0-255)."""
    arr = np.array([[[b * 255, g * 255, r * 255]]], dtype=np.uint8)
    hsl = cv2.cvtColor(arr, cv2.COLOR_BGR2HLS)
    h, l, s = int(hsl[0, 0, 0]), int(hsl[0, 0, 1]), int(hsl[0, 0, 2])
    return h, s, l


def rgb_to_lab(r: float, g: float, b: float) -> tuple[float, float, float]:
    """RGB in [0, 1] -> CIELAB (L: 0-100, a: -128..127, b: -128..127)."""
    arr = np.array([[[b * 255, g * 255, r * 255]]], dtype=np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    return float(lab[0, 0, 0]), float(lab[0, 0, 1]) - 128, float(lab[0, 0, 2]) - 128


def build_color_thresholds(
    rgb: tuple[float, float, float],
    h_tol: int = 15,
    s_min: int = 40,
    s_max: int = 255,
    v_min: int = 40,
    v_max: int = 255,
    l_min: int = 40,
    l_max: int = 200,
    b_star_min: int = 25,
) -> ColorThresholds:
    """Build a ColorThresholds object anchored at the given RGB.

    The HSV range is centred on the anchor hue with `h_tol` tolerance.
    The HSL range mirrors HSV (same hue tolerance, L range).
    The LAB range uses the b* channel (positive for yellow).
    """
    h, s, v = rgb_to_hsv(*rgb)
    h_lo = max(0, h - h_tol)
    h_hi = min(180, h + h_tol)

    _, _, b_star = rgb_to_lab(*rgb)
    lab_lo = np.array([0, 0, max(0, b_star - b_star_min)], dtype=np.uint8)
    lab_hi = np.array([255, 255, 255], dtype=np.uint8)

    return ColorThresholds(
        hsv_lower=np.array([h_lo, s_min, v_min], dtype=np.uint8),
        hsv_upper=np.array([h_hi, s_max, v_max], dtype=np.uint8),
        hsl_lower=np.array([h_lo, l_min, s_min], dtype=np.uint8),
        hsl_upper=np.array([h_hi, l_max, s_max], dtype=np.uint8),
        lab_lower=lab_lo,
        lab_upper=lab_hi,
    )


def build_disease_thresholds(
    rgbs: list[tuple[float, float, float]],
    h_tol: int = 15,
) -> list[ColorThresholds]:
    """Build one ColorThresholds per disease color, for OR-combined masking.

    Each disease (Stressed / Rust / Blight) has a distinct hue range in HSV
    that does not overlap. We anchor one threshold per color and OR them
    inside the detector to capture any diseased crop.
    """
    return [build_color_thresholds(rgb, h_tol=h_tol) for rgb in rgbs]
