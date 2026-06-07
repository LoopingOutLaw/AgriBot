"""Tests for treatment path planning."""
import sys
sys.path.insert(0, '/home/aditya/agribot_ws')

import numpy as np
from src.world_parser import InfectedCrop
from src.treatment_planner import plan_nearest_neighbour, path_length


def make_crop(id_: int, x: float, y: float) -> InfectedCrop:
    return InfectedCrop(id=id_, gz_x=x, gz_y=y, gz_z=0.4)


def test_empty_crops_returns_empty():
    assert plan_nearest_neighbour(np.array([0.0, 0.0]), []) == []


def test_single_crop():
    c = make_crop(1, 1.0, 1.0)
    result = plan_nearest_neighbour(np.array([0.0, 0.0]), [c])
    assert len(result) == 1
    assert result[0].id == 1


def test_first_crop_is_closest_to_start():
    crops = [
        make_crop(1, 5.0, 5.0),    # far
        make_crop(2, 0.5, 0.5),    # close
        make_crop(3, 3.0, 3.0),    # mid
    ]
    result = plan_nearest_neighbour(np.array([0.0, 0.0]), crops)
    assert result[0].id == 2


def test_all_crops_visited_exactly_once():
    crops = [make_crop(i, float(i), float(i)) for i in range(1, 9)]
    result = plan_nearest_neighbour(np.array([0.0, 0.0]), crops)
    assert len(result) == 8
    assert {c.id for c in result} == {1, 2, 3, 4, 5, 6, 7, 8}


def test_greedy_path_chooses_reasonable_order():
    """With crops along a line, the path should not backtrack."""
    crops = [
        make_crop(1, 0.0, 0.0),
        make_crop(2, 0.0, 2.0),
        make_crop(3, 0.0, 4.0),
    ]
    result = plan_nearest_neighbour(np.array([0.0, 0.0]), crops)
    # Should be (0,0) -> (0,2) -> (0,4) or reverse
    ys = [c.gz_y for c in result]
    assert ys == sorted(ys) or ys == sorted(ys, reverse=True)


def test_path_length_calculates_correctly():
    crops = [make_crop(1, 0.0, 0.0), make_crop(2, 3.0, 4.0)]
    length = path_length(np.array([0.0, 0.0]), crops)
    assert abs(length - 5.0) < 1e-9  # 0->(0,0)=0, (0,0)->(3,4)=5
