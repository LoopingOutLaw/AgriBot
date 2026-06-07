"""Empirical camera-axis calibration.

The gz camera-axis convention is not documented cleanly. We recover the
correct mapping (sign flips for image u and v) by trying all four sign
combinations and selecting the one that places the camera-projected point
closest to a known world position.
"""
from dataclasses import dataclass
import math
import numpy as np

from src.camera_math import (
    CameraIntrinsics,
    euler_to_rotation,
    pixel_to_ray,
    ray_plane_intersect,
)


@dataclass(frozen=True)
class AxisMapping:
    sign_u: int   # +1 or -1: maps image-u to camera-frame x
    sign_v: int   # +1 or -1: maps image-v to camera-frame y
    error: float = 0.0  # calibration error in metres at the anchor


def _mount_rotation(intr: CameraIntrinsics) -> np.ndarray:
    """Rotation that takes camera-local rays to body frame."""
    return euler_to_rotation(0, intr.mount_pitch, 0)


def _build_world_ray(
    pixel: tuple[int, int],
    drone_pos: np.ndarray,
    drone_att: tuple[float, float, float],
    intr: CameraIntrinsics,
    sign_u: int,
    sign_v: int,
) -> np.ndarray:
    u, v = pixel
    # Apply sign flips
    u_flipped = (intr.width - 1 - u) if sign_u == -1 else u
    v_flipped = (intr.height - 1 - v) if sign_v == -1 else v
    ray_cam = pixel_to_ray(u_flipped, v_flipped, intr)
    R_mount = _mount_rotation(intr)
    R_body = euler_to_rotation(*drone_att)
    R = R_body @ R_mount
    return R @ ray_cam


def calibrate_axis_mapping(
    correspondences: list[tuple[np.ndarray, tuple[float, float, float], tuple[int, int], np.ndarray]],
    intr: CameraIntrinsics,
    max_error_m: float = 0.5,
) -> AxisMapping | None:
    """Try all 4 sign combinations; return the best with error < max_error_m.

    Each correspondence is (drone_pos, drone_att, pixel, known_world_point).
    """
    best: AxisMapping | None = None
    for sign_u in (1, -1):
        for sign_v in (1, -1):
            errors = []
            for drone_pos, drone_att, pixel, known_pt in correspondences:
                cam_pos = drone_pos + np.array([0.0, 0.0, -0.05])
                ray_world = _build_world_ray(
                    pixel, drone_pos, drone_att, intr, sign_u, sign_v
                )
                hit = ray_plane_intersect(cam_pos, ray_world, plane_z=known_pt[2])
                if hit is None:
                    errors.append(float("inf"))
                else:
                    err = float(np.linalg.norm(hit[:2] - known_pt[:2]))
                    errors.append(err)
            mean_err = float(np.mean(errors))
            if best is None or mean_err < best.error:
                best = AxisMapping(sign_u=sign_u, sign_v=sign_v, error=mean_err)
    if best is None or best.error > max_error_m:
        return None
    return best
