"""Pinhole camera math: intrinsics, pixel-to-ray, ray-plane intersection."""
import math
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    hfov: float         # radians
    mount_pitch: float  # radians, mount rotation about the drone's Y axis


def derive_vfov(hfov: float, width: int, height: int) -> float:
    """Derive VFOV from HFOV and image aspect ratio."""
    return 2.0 * math.atan((height / width) * math.tan(hfov / 2.0))


def pixel_to_ray(u: float, v: float, intr: CameraIntrinsics) -> np.ndarray:
    """Convert pixel (u, v) to a unit ray in the camera's local frame.

    Camera convention (this module): the camera's FORWARD is the +X axis
    of the local frame. Image +u (right) maps to camera +Y (right).
    Image +v (down) maps to camera +Z (down).
    A pixel above the centre (small v) gives a ray with NEGATIVE z in the
    camera frame.

    Note: this is the gz camera convention where the lens's optical axis
    is the camera's local +X. After applying a mount pitch of pi/2 (R_y),
    +X maps to -Z (down), so the camera looks straight down.
    """
    vfov = derive_vfov(intr.hfov, intr.width, intr.height)
    fx = intr.width / (2.0 * math.tan(intr.hfov / 2.0))
    fy = intr.height / (2.0 * math.tan(vfov / 2.0))
    cx, cy = intr.width / 2.0, intr.height / 2.0
    # Forward is +X in the camera's local frame (NOT +Z)
    fwd = 1.0
    y_cam = (u - cx) / fx  # image right -> camera +Y
    z_cam = (v - cy) / fy  # image down -> camera +Z
    ray = np.array([fwd, y_cam, z_cam], dtype=float)
    ray /= np.linalg.norm(ray)
    return ray


def ray_plane_intersect(
    cam_pos: np.ndarray, ray_world: np.ndarray, plane_z: float
) -> np.ndarray | None:
    """Intersect a world-frame ray with a horizontal plane at plane_z.

    Returns the (x, y, z) hit point, or None if the ray is parallel to the
    plane or points away from it.
    """
    if abs(ray_world[2]) < 1e-9:
        return None
    t = (plane_z - cam_pos[2]) / ray_world[2]
    if t < 0:
        return None  # ray going the wrong way
    return cam_pos + t * ray_world


def euler_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from roll, pitch, yaw (radians).

    Convention: gz/ROS body frame is ENU with +X forward, +Y left, +Z up.
    Rotation order is roll (about X), then pitch (about Y), then yaw (about Z):
        R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return R_z @ R_y @ R_x
