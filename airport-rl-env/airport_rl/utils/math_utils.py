"""Small numeric helpers shared across the project."""

from __future__ import annotations

import numpy as np


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def quat_from_yaw(yaw: float) -> np.ndarray:
    """Quaternion (w, x, y, z) for a pure rotation about Z by ``yaw``."""
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def yaw_from_quat(quat: np.ndarray) -> float:
    """Extract the yaw (rotation about Z) from a (w, x, y, z) quaternion."""
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def relative_polar(dx: float, dy: float, yaw: float) -> tuple[float, float]:
    """Return (distance, bearing) of a point (dx, dy) relative to the agent.

    Bearing is the angle the agent would have to turn (in its own frame) to
    face the point, wrapped to (-pi, pi].
    """
    dist = float(np.hypot(dx, dy))
    bearing = wrap_angle(np.arctan2(dy, dx) - yaw)
    return dist, bearing


def lidar_directions(n_rays: int) -> np.ndarray:
    """Unit XY vectors for ``n_rays`` evenly spaced around a full circle."""
    angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)
