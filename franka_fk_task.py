"""Run Panda forward kinematics for the angles in joint_inputs.py.

The Panda model supplies home geometry and a MuJoCo reference pose.
fk_algorithm.py computes the hand pose independently from that geometry.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

import fk_algorithm
import joint_inputs
import panda_model


POSITION_TOLERANCE: float = 1e-6  # metres
ORIENTATION_TOLERANCE: float = 1e-6  # radians


def fk_mujoco(q: ArrayLike) -> NDArray[np.float64]:
    """Get the reference hand pose from MuJoCo at the requested angles."""
    return panda_model.hand_transform_at(fk_algorithm.validate_joint_angles(q))


def fk_manual(q: ArrayLike) -> NDArray[np.float64]:
    """Calculate the hand pose from the Panda model's home geometry."""
    return fk_algorithm.fk_manual(
        q,
        panda_model.HOME_Q,
        panda_model.HOME_AXES,
        panda_model.HOME_ANCHORS,
        panda_model.HOME_HAND_TRANSFORM,
    )


def compare_fk(q: ArrayLike, label: str) -> tuple[float, float]:
    """Compare both hand poses; report distance in metres and angle in radians."""
    q = fk_algorithm.validate_joint_angles(q)
    expected = fk_mujoco(q)
    calculated = fk_manual(q)

    position_error = float(
        np.linalg.norm(expected[:3, 3] - calculated[:3, 3])
    )
    orientation_error = fk_algorithm.rotation_error_angle(
        expected[:3, :3], calculated[:3, :3]
    )

    print(f"\n{label}")
    print("q:", np.round(q, 4))
    print("MuJoCo position:   ", np.round(expected[:3, 3], 6))
    print("Manual FK position:", np.round(calculated[:3, 3], 6))
    print(f"Position error:    {position_error:.3e} m")
    print(f"Orientation error: {orientation_error:.3e} rad")

    return position_error, orientation_error


def main() -> None:
    np.set_printoptions(precision=5, suppress=True)

    print(f"Model: {panda_model.XML_PATH}")
    print("qpos indices:", panda_model.qpos_indices)
    print("DoF indices: ", panda_model.dof_indices)
    print("Home joint angles:", panda_model.HOME_Q)
    print("\nHome hand transformation:")
    print(panda_model.HOME_HAND_TRANSFORM)

    joint_min, joint_max = panda_model.joint_limits()
    test_results = [
        compare_fk(q, label)
        for label, q in joint_inputs.test_configurations(
            panda_model.HOME_Q, joint_min, joint_max
        )
    ]

    worst_position_error = max(result[0] for result in test_results)
    worst_orientation_error = max(result[1] for result in test_results)

    print("\nFinal result")
    print(f"Worst position error:    {worst_position_error:.3e} m")
    print(f"Worst orientation error: {worst_orientation_error:.3e} rad")

    passed = (
        worst_position_error < POSITION_TOLERANCE
        and worst_orientation_error < ORIENTATION_TOLERANCE
    )
    if passed:
        print("PASS: Manual FK agrees with MuJoCo.")
    else:
        print("FAIL: Manual FK does not agree with MuJoCo.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
