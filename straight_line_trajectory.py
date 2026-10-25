"""World-space straight position path and shortest-arc SO(3) SLERP.

No robot, MuJoCo, task input, or viewer is loaded here. Rotation-matrix
exponentiation implements the same shortest arc as unit-quaternion SLERP.
"""

import numpy as np

from FK_algorithm import skew
from IK_algorithm import rotation_log
from jacobian_algorithm import transform_inverse


def create_trajectory(start, goal, duration=5.0):
    """Factory used by trajectory_config; the reference mathematics is shared."""
    return StraightLineTrajectory(start, goal, duration)


class StraightLineTrajectory:
    """Synchronize translation and rotation with cubic time scaling.

The tracked point is the origin of the supplied end-effector frame.
At exactly 180 degrees either rotation direction is a shortest arc; the
existing rotation_log convention selects one consistently for this instance.
"""

    def __init__(self, start, goal, duration=5.0):
        transform_inverse(start)
        transform_inverse(goal)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be positive and finite")
        self.start = np.array(start, dtype=float, copy=True)
        self.goal = np.array(goal, dtype=float, copy=True)
        self.duration = float(duration)
        self._translation = self.goal[:3, 3] - self.start[:3, 3]
        self._rotation_vector = rotation_log(self.start[:3, :3].T @ self.goal[:3, :3])
        self.start.setflags(write=False)
        self.goal.setflags(write=False)

    def sample(self, time):
        """Return (desired SE(3) pose, desired-body twist [omega; v]).

Times outside [0, duration] hold the respective endpoint with zero velocity.
The linear twist component is R_d.T @ p_dot, not world-frame p_dot.
"""
        if not np.isfinite(time):
            raise ValueError("sample time must be finite")
        u = float(np.clip(time / self.duration, 0.0, 1.0))
        s = u * u * (3 - 2 * u)
        s_dot = 6 * u * (1 - u) / self.duration
        rotation_vector = s * self._rotation_vector
        angle = np.linalg.norm(rotation_vector)
        cross = skew(rotation_vector)
        # Stable Rodrigues coefficients, including exactly zero rotation.
        rotation = (np.eye(3) + np.sinc(angle / np.pi) * cross
                    + 0.5 * np.sinc(angle / (2 * np.pi))**2 * cross @ cross)
        desired = np.eye(4)
        desired[:3, :3] = self.start[:3, :3] @ rotation
        desired[:3, 3] = self.start[:3, 3] + s * self._translation
        twist = np.r_[s_dot * self._rotation_vector,
                      desired[:3, :3].T @ (s_dot * self._translation)]
        return desired, twist

    def sample_times(self, dt=0.01, settle_time=1.0):
        """Include t=0, the exact arrival time, and an optional goal-hold tail.

The last interval in either segment can be shorter than dt.
"""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if not np.isfinite(settle_time) or settle_time < 0:
            raise ValueError("settle_time must be nonnegative and finite")
        times = np.r_[np.arange(0.0, self.duration, dt), self.duration]
        if settle_time > 0:
            times = np.r_[times, self.duration + np.arange(dt, settle_time, dt),
                          self.duration + settle_time]
        return np.unique(times)
