"""Run FK using the selected profile's robot-specific joint inputs.

robot_config.py selects the model; FK_algorithm.py computes the endpoint pose
independently from the captured home geometry.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

import FK_algorithm
import joint_inputs
from robot_config import robot


POSITION_TOLERANCE: float = 1e-6  # metres
ORIENTATION_TOLERANCE: float = 1e-6  # radians


def fk_mujoco(q: ArrayLike) -> NDArray[np.float64]:
    """Get the reference hand pose from MuJoCo at the requested angles."""
    return robot.transform_at(FK_algorithm.validate_joint_angles(q, robot.joint_count))


def fk_manual(q: ArrayLike) -> NDArray[np.float64]:
    """Calculate the selected end-effector pose from home geometry."""
    return FK_algorithm.fk_manual(
        q,
        robot.home_q,
        robot.home_axes,
        robot.home_anchors,
        robot.home_transform,
        robot.joint_types,
    )


def compare_fk(q: ArrayLike, label: str) -> tuple[float, float]:
    """Compare both hand poses; report distance in metres and angle in radians."""
    q = FK_algorithm.validate_joint_angles(q, robot.joint_count)
    expected = fk_mujoco(q)
    calculated = fk_manual(q)

    position_error = float(
        np.linalg.norm(expected[:3, 3] - calculated[:3, 3])
    )
    orientation_error = FK_algorithm.rotation_error_angle(
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

    print(f"Model: {robot.xml_path}")
    print(f"Arm joints ({robot.joint_count}): {robot.joint_names}")
    print("qpos indices:", robot.qpos_indices)
    print("DoF indices: ", robot.dof_indices)
    print("Home joint coordinates:", robot.home_q)
    print("\nHome end-effector transformation:")
    print(robot.home_transform)

    test_results = [
        compare_fk(q, label)
        for label, q in joint_inputs.test_configurations(robot)
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
