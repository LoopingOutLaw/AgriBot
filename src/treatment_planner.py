"""Nearest-neighbour path planning over infected crops."""
import numpy as np

from src.world_parser import InfectedCrop


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def plan_nearest_neighbour(
    start: np.ndarray, crops: list[InfectedCrop]
) -> list[InfectedCrop]:
    """Return the crops in visit order using a greedy nearest-neighbour heuristic."""
    if not crops:
        return []
    remaining = list(crops)
    ordered: list[InfectedCrop] = []
    current = np.array(start, dtype=float)
    while remaining:
        nearest = min(
            remaining,
            key=lambda c: _dist(current, np.array([c.gz_x, c.gz_y])),
        )
        ordered.append(nearest)
        current = np.array([nearest.gz_x, nearest.gz_y], dtype=float)
        remaining.remove(nearest)
    return ordered


def path_length(start: np.ndarray, crops: list[InfectedCrop]) -> float:
    """Total Euclidean path length through the given crop order."""
    if not crops:
        return 0.0
    total = _dist(np.array(start, dtype=float), np.array([crops[0].gz_x, crops[0].gz_y]))
    for a, b in zip(crops, crops[1:]):
        total += _dist(np.array([a.gz_x, a.gz_y]), np.array([b.gz_x, b.gz_y]))
    return total
