"""Shared IK target for the configured end-effector body, in world coordinates.

Position: metres. Orientation: roll, pitch, yaw in degrees.
R = Rz(yaw) @ Ry(pitch) @ Rx(roll), mapping endpoint coordinates into world.
These are absolute pose coordinates, not offsets from the home pose.
Save changes and restart franka_ik_GUI.py to apply them.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


TARGET_POSITION = [0.5, 0.56, 0.45]  # x, y, z (m)
TARGET_RPY_DEG = [130.0, 45.0, 70.0]   # roll, pitch, yaw (degrees)

# None starts from the selected robot's home pose. Otherwise supply exactly
# n absolute joint coordinates (rad for hinges, m for slides), not offsets.
INITIAL_JOINTS = None


def target_transform(
    position: ArrayLike | None = None, rpy_deg: ArrayLike | None = None,
) -> NDArray[np.float64]:
    position = np.asarray(TARGET_POSITION if position is None else position, dtype=float)
    angles = np.asarray(TARGET_RPY_DEG if rpy_deg is None else rpy_deg, dtype=float)
    if position.shape != (3,) or angles.shape != (3,):
        raise ValueError("Position and roll/pitch/yaw must each contain three numbers")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(angles)):
        raise ValueError("Position and orientation must be finite")
    roll, pitch, yaw = np.deg2rad(angles)
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = position
    return transform
