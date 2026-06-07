"""Color table for the 4 disease types and helper to classify from RGB."""
from dataclasses import dataclass
from enum import Enum


class DiseaseType(str, Enum):
    HEALTHY = "Healthy"
    STRESSED = "Stressed"
    RUST = "Rust"
    BLIGHT = "Blight"


# RGB tuples in the same (R, G, B) order as the SDF <material><ambient> tags.
DISEASE_COLORS: dict[str, tuple[float, float, float]] = {
    "Healthy":  (0.1, 0.6, 0.1),
    "Stressed": (0.4, 0.6, 0.2),
    "Rust":     (0.7, 0.4, 0.1),
    "Blight":   (0.5, 0.1, 0.5),  # magenta (H≈150) - far from other hues
}


def color_to_disease(r: float, g: float, b: float) -> str:
    """Return the disease name whose reference color is nearest to (r, g, b)."""
    best_name = "Healthy"
    best_dist = float("inf")
    for name, (cr, cg, cb) in DISEASE_COLORS.items():
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name
