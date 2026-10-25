"""Joint-coordinate inputs sized by the selected robot's serial chain.

Edit the input module named by the selected profile's FK_INPUT_MODULE.
load_offsets(profile) reads the profile's FK_INPUT_MODULE only when requested.
Callers pass these task inputs separately from RobotModel. Coordinates are
radians for revolute joints and metres for prismatic joints; offsets are
relative to home. Omitting offsets in the numeric helpers selects home.
"""

from collections.abc import Mapping
import importlib
from types import ModuleType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from FK_algorithm import validate_joint_angles
from robot_model import RobotModel


RANDOM_TEST_COUNT = 10
RANDOM_SEED = 7


def load_offsets(profile: ModuleType) -> dict[str, float]:
    """Read a copy of the selected profile's named FK offsets, only for FK tasks.

    Joint names, values, and limits are checked against the model by
    selected_joint_angles(), rather than during model construction.
    """
    inputs = importlib.import_module(profile.FK_INPUT_MODULE)
    if not isinstance(inputs.JOINT_OFFSETS, Mapping):
        raise ValueError("JOINT_OFFSETS must map joint names to offsets")
    return dict(inputs.JOINT_OFFSETS)


def selected_joint_angles(
    robot: RobotModel,
    offsets: ArrayLike | Mapping[str, float] | None = None,
) -> NDArray[np.float64]:
    if offsets is None:
        offsets = {}
    if isinstance(offsets, Mapping):
        unknown = offsets.keys() - set(robot.joint_names)
        if unknown:
            raise ValueError(f"Offsets refer to joints outside the arm: {sorted(unknown)}")
        offsets = [offsets.get(name, 0.0) for name in robot.joint_names]
    offsets = validate_joint_angles(offsets, robot.joint_count)
    chosen_q = robot.home_q + offsets
    lower, upper = robot.joint_limits()
    outside = np.flatnonzero((chosen_q < lower) | (chosen_q > upper))
    if outside.size:
        names = ", ".join(robot.joint_names[index] for index in outside)
        raise ValueError(f"Chosen coordinates exceed joint limits: {names}")
    return chosen_q


def test_configurations(
    robot: RobotModel,
    offsets: ArrayLike | Mapping[str, float] | None = None,
) -> list[tuple[str, NDArray[np.float64]]]:
    cases = [
        ("Test 1: home configuration", robot.home_q.copy()),
        ("Test 2: chosen joint inputs", selected_joint_angles(robot, offsets)),
    ]
    lower, upper = robot.joint_limits()
    # Finite sampling windows for unlimited joints, not physical limits.
    span = np.array([0.5 if kind == "revolute" else 0.05 for kind in robot.joint_types])
    lower = np.where(np.isfinite(lower), lower, robot.home_q - span)
    upper = np.where(np.isfinite(upper), upper, robot.home_q + span)
    random_generator = np.random.default_rng(RANDOM_SEED)
    for index in range(RANDOM_TEST_COUNT):
        cases.append((f"Random test {index + 1}", random_generator.uniform(lower, upper)))
    return cases
