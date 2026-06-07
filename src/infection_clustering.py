"""Cluster infection positions to find disease hotspots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.cluster import DBSCAN, KMeans


@dataclass
class ClusterResult:
    method: str
    labels: np.ndarray            # cluster id per point; -1 = noise (DBSCAN only)
    centers: list[tuple[float, float]]


def cluster_infections(
    positions: list[tuple[float, float]],
    method: Literal["kmeans", "dbscan"] = "kmeans",
    k: int = 3,
    eps: float = 0.7,
    min_samples: int = 2,
) -> ClusterResult:
    """Cluster infection positions in gz coordinates."""
    if not positions:
        return ClusterResult(method=method, labels=np.array([], dtype=int), centers=[])

    X = np.array(positions, dtype=float)

    if method == "kmeans":
        k_eff = min(k, len(positions))
        model = KMeans(n_clusters=k_eff, n_init=10, random_state=42)
        labels = model.fit_predict(X)
        centers = [(float(c[0]), float(c[1])) for c in model.cluster_centers_]
        return ClusterResult(method="kmeans", labels=labels, centers=centers)

    if method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        unique = sorted(set(labels.tolist()) - {-1})
        centers = []
        for cid in unique:
            pts = X[labels == cid]
            centers.append((float(pts[:, 0].mean()), float(pts[:, 1].mean())))
        return ClusterResult(method="dbscan", labels=labels, centers=centers)

    raise ValueError(f"unknown method: {method}")
