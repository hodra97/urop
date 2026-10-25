"""Display and verify space/body Jacobians for the selected robot.

Target offsets come from the selected profile's robot-specific input file.
Run: .venv/bin/python jacobian.py
No viewer or simulation stepping is required. The analytical algorithm uses
home geometry only; MuJoCo supplies an independent verification reference.
"""

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

import FK_algorithm
import jacobian_algorithm
import joint_inputs
import robot_config
from robot_model import RobotModel


JACOBIAN_TOLERANCE: float = 1e-9


def jacobians_manual(robot: RobotModel, q: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    """Return (space, body) Jacobians, both in angular-first order."""
    geometry = (
        q, robot.home_q, robot.home_axes, robot.home_anchors
    )
    space = jacobian_algorithm.jacobian_space(*geometry, robot.joint_types)
    body = jacobian_algorithm.jacobian_body(*geometry, robot.home_transform, robot.joint_types)
    return space, body


def jacobians_mujoco(robot: RobotModel, q: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    """Convert MuJoCo's endpoint derivatives into space/body twists."""
    transform = robot.transform_at(FK_algorithm.validate_joint_angles(q, robot.joint_count))
    jacp = np.zeros((3, robot.model.nv))
    jacr = np.zeros((3, robot.model.nv))
    mujoco.mj_jacBody(
        robot.model, robot.data, jacp, jacr, robot.end_effector_body_id
    )
    # qpos indices and velocity DoF indices are different concepts.
    angular = jacr[:, robot.dof_indices]
    linear = jacp[:, robot.dof_indices]
    rotation = transform[:3, :3]
    position = transform[:3, 3]

    # These conversions deliberately do not call the analytical helpers.
    space = np.vstack((angular, linear + np.cross(position, angular.T).T))
    body = np.vstack((rotation.T @ angular, rotation.T @ linear))
    return space, body


def run_checks(robot: RobotModel, offsets=None) -> None:
    """Verify the supplied FK task and random poses against MuJoCo."""
    np.set_printoptions(precision=6, suppress=True)
    chosen_q = joint_inputs.selected_joint_angles(robot, offsets)
    labels = ("J_s: space twist", "J_b: end-effector-frame twist")

    print("Home q:", robot.home_q)
    print(f"Arm joints ({robot.joint_count}): {robot.joint_names}")
    print("Chosen q:", chosen_q)
    print("theta = q - home_q:", chosen_q - robot.home_q)
    for label, matrix in zip(labels, jacobians_manual(robot, chosen_q)):
        print(f"\n{label} ({matrix.shape[0]} x {matrix.shape[1]})")
        print(matrix)

    worst = np.zeros(len(labels))
    cases = joint_inputs.test_configurations(robot, offsets)
    for label, q in cases:
        manual = jacobians_manual(robot, q)
        reference = jacobians_mujoco(robot, q)
        errors = np.array([
            np.max(np.abs(actual - expected))
            for actual, expected in zip(manual, reference)
        ])
        worst = np.maximum(worst, errors)
        print(f"\n{label}: max |difference| for J_s, J_b = {errors.tolist()}")

    print(f"\nWorst errors over {len(cases)} poses: {worst.tolist()}")
    if not np.all(worst < JACOBIAN_TOLERANCE):
        raise SystemExit("FAIL: analytical Jacobians disagree with MuJoCo.")
    print("PASS: space and body Jacobians agree with MuJoCo.")


def main() -> None:
    profile = robot_config.load_profile()
    robot = robot_config.load_robot(profile)
    run_checks(robot, joint_inputs.load_offsets(profile))


if __name__ == "__main__":
    main()
