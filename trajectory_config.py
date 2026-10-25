"""Explicit trajectory selection, independent of ROBOT_MODULE and MuJoCo.

TRAJECTORY_MODULE is read at call time. A module exports
create_trajectory(start, goal, duration), returning an object with start, goal,
duration, and sample(tau) -> (world SE(3) pose, desired-body [omega; v] twist).
Twist is differentiated with respect to nominal trajectory time tau, not s.
"""

import importlib
import os

import numpy as np

from IK_algorithm import pose_error
from jacobian_algorithm import transform_inverse


DEFAULT_TRAJECTORY_MODULE = "straight_line_trajectory"


def load_profile(module_name=None):
    """Import a trajectory module without constructing any reference or robot."""
    if module_name is None:
        module_name = os.environ.get("TRAJECTORY_MODULE", DEFAULT_TRAJECTORY_MODULE)
    return importlib.import_module(module_name)


def sample_reference(trajectory, tau):
    """Validate the plugin boundary and return copied pose/twist arrays."""
    pose, twist = trajectory.sample(float(tau))
    pose, twist = np.array(pose, dtype=float, copy=True), np.array(twist, dtype=float, copy=True)
    transform_inverse(pose)
    if twist.shape != (6,) or not np.all(np.isfinite(twist)):
        raise ValueError("Trajectory sample twist must be a finite six-vector [omega; v]")
    return pose, twist


def validate_trajectory(trajectory):
    """Validate endpoint metadata and sample contract; no concrete class required."""
    if not all(hasattr(trajectory, name) for name in ("start", "goal", "duration", "sample")):
        raise ValueError("Trajectory needs start, goal, duration, and sample(tau)")
    if not callable(trajectory.sample):
        raise ValueError("Trajectory sample must be callable")
    if not np.isfinite(trajectory.duration) or trajectory.duration <= 0:
        raise ValueError("Trajectory duration must be positive and finite")
    for tau, expected in ((0.0, trajectory.start), (trajectory.duration, trajectory.goal)):
        pose, _ = sample_reference(trajectory, tau)
        _, position, rotation = pose_error(pose, expected)
        if position > 1e-8 or rotation > 1e-8:
            raise ValueError("Trajectory samples must agree with its start and goal")
    return trajectory


def load_trajectory(start, goal, duration=5.0, profile=None):
    """Create a fresh reference from the selected module at execution time."""
    if profile is None:
        profile = load_profile()
    factory = getattr(profile, "create_trajectory", None)
    if not callable(factory):
        raise ValueError(f"{profile.__name__} must export create_trajectory(start, goal, duration)")
    trajectory = validate_trajectory(factory(np.array(start, copy=True), np.array(goal, copy=True), duration))
    if not np.isclose(trajectory.duration, duration, rtol=0, atol=1e-12):
        raise ValueError("Trajectory factory must preserve the requested duration")
    for actual, expected in ((trajectory.start, start), (trajectory.goal, goal)):
        _, position, rotation = pose_error(actual, expected)
        if position > 1e-8 or rotation > 1e-8:
            raise ValueError("Trajectory factory must preserve the requested endpoints")
    return trajectory
