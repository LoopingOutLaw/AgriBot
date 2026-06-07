"""Tests for infection clustering (K-means + DBSCAN)."""
import sys
sys.path.insert(0, '/home/aditya/agribot_ws')

import numpy as np
import pytest

from src.infection_clustering import cluster_infections, ClusterResult


def test_kmeans_returns_3_clusters_for_3_groups():
    positions = [
        (0.0, 0.0), (0.1, 0.1), (-0.1, 0.05),       # cluster 1
        (5.0, 5.0), (5.1, 5.05), (4.9, 4.95),       # cluster 2
        (10.0, 0.0), (10.05, 0.1), (9.95, -0.05),   # cluster 3
    ]
    res = cluster_infections(positions, method="kmeans", k=3)
    assert res.method == "kmeans"
    assert len(res.centers) == 3
    assert set(res.labels.tolist()) == {0, 1, 2}


def test_dbscan_labels_outliers_as_minus_one():
    positions = [
        (0.0, 0.0), (0.1, 0.05),  # cluster
        (5.0, 5.0),               # outlier
    ]
    res = cluster_infections(positions, method="dbscan", eps=0.5, min_samples=2)
    assert -1 in res.labels.tolist()


def test_empty_input_returns_empty_result():
    res = cluster_infections([], method="kmeans", k=3)
    assert res.method == "kmeans"
    assert len(res.centers) == 0
    assert len(res.labels) == 0


def test_single_point_returns_one_cluster():
    res = cluster_infections([(1.0, 2.0)], method="kmeans", k=1)
    assert len(res.centers) == 1
    assert res.centers[0] == (1.0, 2.0)


def test_cluster_centers_are_in_gz_units():
    positions = [(2.5, 3.7), (2.6, 3.6), (2.4, 3.8)]
    res = cluster_infections(positions, method="kmeans", k=1)
    cx, cy = res.centers[0]
    assert 2.0 < cx < 3.0
    assert 3.0 < cy < 4.0
