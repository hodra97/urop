"""Joint-angle inputs for the Franka forward-kinematics examples.

Edit JOINT_OFFSETS_RAD below to change the pose tested by franka_fk_task.py
and displayed by franka_fk_task_visualized.py. Each chosen joint angle is
HOME_Q + its offset; for example, joint4 has a nonzero home angle, so an
offset of -0.10 rad does not mean its final angle is -0.10 rad.

The FK formula lives in fk_algorithm.py and does not need to change when
you choose new angles. panda_model.py loads the robot geometry. These offsets
select a posture, not a timed path.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


# The seven values correspond to joint1, joint2, ..., joint7, in radians.
# They are added to the model's "home" joint angles (they are not absolute
# angles). A zero offset leaves that joint at its home angle. Positive and
# negative signs follow the axes defined by the Panda model.
JOINT_OFFSETS_RAD: tuple[float, ...] = (
    0.20,  # joint1
    -0.10,  # joint2
    0.15,  # joint3
    -0.10,  # joint4
    0.10,  # joint5
    0.15,  # joint6
    -0.20,  # joint7
)

# Extra random configurations keep the original FK-versus-MuJoCo check.
# Set RANDOM_TEST_COUNT to 0 if you only want to check home and your pose.
# The fixed seed makes the same random test poses appear on every run.
RANDOM_TEST_COUNT: int = 10
RANDOM_SEED: int = 7


def selected_joint_angles(
    home_q: ArrayLike,
    joint_min: ArrayLike,
    joint_max: ArrayLike,
) -> NDArray[np.float64]:
    """Compute q_selected = q_home + offsets, then check joint limits.

    Inputs are seven-element arrays in joint1-to-joint7 order. This function
    rejects an invalid pose instead of silently clipping its angles.
    """
    home = np.asarray(home_q, dtype=float)
    lower = np.asarray(joint_min, dtype=float)
    upper = np.asarray(joint_max, dtype=float)
    offsets = np.asarray(JOINT_OFFSETS_RAD, dtype=float)

    if any(values.shape != (7,) for values in (home, lower, upper, offsets)):
        raise ValueError("Home, limits, and offsets must each have 7 values.")

    # Elementwise addition changes each joint independently from home.
    chosen_q = home + offsets
    if not np.all(np.isfinite(chosen_q)):
        raise ValueError("All joint angles must be finite numbers.")

    outside_limits = np.flatnonzero((chosen_q < lower) | (chosen_q > upper))
    if outside_limits.size:
        names = ", ".join(f"joint{index + 1}" for index in outside_limits)
        raise ValueError(f"Chosen angles exceed the limits of {names}.")

    return chosen_q


def test_configurations(
    home_q: ArrayLike,
    joint_min: ArrayLike,
    joint_max: ArrayLike,
) -> list[tuple[str, NDArray[np.float64]]]:
    """Return (label, seven joint angles) pairs for the FK checks."""
    home = np.asarray(home_q, dtype=float)
    lower = np.asarray(joint_min, dtype=float)
    upper = np.asarray(joint_max, dtype=float)

    cases = [
        ("Test 1: home configuration", home.copy()),
        ("Test 2: chosen joint inputs", selected_joint_angles(home, lower, upper)),
    ]

    # Random angles cover more of each joint's legal range than the one
    # user-selected pose and help catch mistakes in the FK implementation.
    random_generator = np.random.default_rng(seed=RANDOM_SEED)
    for test_number in range(RANDOM_TEST_COUNT):
        random_q = random_generator.uniform(lower, upper)
        cases.append((f"Random test {test_number + 1}", random_q))

    return cases
