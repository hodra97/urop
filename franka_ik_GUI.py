"""Solve a world-frame pose target, print joint coordinates, and visualize it.

ROBOT_MODULE selects the robot. Edit end_effector_inputs.py for the common
target; --no-viewer performs the same calculation without a graphical desktop.
The animation interpolates joint coordinates to the solution and holds there.
It is not actuator control, a Cartesian trajectory, or collision avoidance.
"""

import argparse
from dataclasses import replace
import time

import mujoco
import mujoco.viewer
import numpy as np

import end_effector_inputs
from FK_algorithm import validate_joint_angles
from IK_algorithm import IKResult, pose_error, solve_ik
from robot_config import robot
from viewer_overlays import desired_pose_text, draw_target_axes, draw_world_grid


def solve_target(target, initial, *, max_iterations=250, restarts=8) -> IKResult:
    lower, upper = robot.joint_limits()
    result = solve_ik(
        target, initial, robot.home_q, robot.home_axes, robot.home_anchors,
        robot.home_transform, robot.joint_types, lower_limits=lower, upper_limits=upper,
        max_iterations=max_iterations, restarts=restarts,
    )
    if result.success:
        # Independently verify the solution against the selected MuJoCo model.
        _, position_error, orientation_error = pose_error(robot.transform_at(result.q), target)
        if position_error > 1e-4 or orientation_error > 1e-3:
            return replace(result, success=False, status="VERIFICATION_FAILED",
                           message="MuJoCo pose does not meet the IK tolerances",
                           position_error=position_error, orientation_error=orientation_error)
    return result


def print_result(result: IKResult) -> None:
    print(f"\n{result.status}: {result.message}")
    print(f"Position error: {result.position_error:.6e} m")
    print(f"Orientation error: {result.orientation_error:.6e} rad")
    print(f"Iterations: {result.iterations}; attempts: {result.attempts}")
    print("Final joint coordinates:" if result.success else "Best candidate (not a valid IK solution):")
    print(f"{'Joint':24s} {'rad / m':>12s} {'degrees':>12s}")
    for name, kind, value in zip(robot.joint_names, robot.joint_types, result.q):
        degrees = f"{np.rad2deg(value):12.4f}" if kind == "revolute" else f"{'--':>12s}"
        print(f"{name:24s} {value:12.6f} {degrees}")


def draw_target(scene, target) -> None:
    """Target endpoint axes: x red, y green, z blue, fixed in world space."""
    draw_target_axes(scene, target)


def visualize_solution(target, initial, result: IKResult, travel_time: float = 3.0,
                       *, target_rpy_deg=None) -> None:
    if not np.isfinite(travel_time) or travel_time <= 0:
        raise ValueError("Animation duration must be positive and finite")
    initial = validate_joint_angles(initial, robot.joint_count).copy()
    # Failure leaves the robot at the initial posture, never at a false solution.
    goal = result.q if result.success else initial
    robot.reset()
    robot.transform_at(initial)
    target_text = desired_pose_text(target, target_rpy_deg)
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            draw_target(viewer.user_scn, target)
            draw_world_grid(viewer.user_scn)
        start_time = time.perf_counter()
        while viewer.is_running():
            frame_start = time.perf_counter()
            fraction = np.clip((frame_start - start_time) / travel_time, 0.0, 1.0)
            blend = fraction**2 * (3 - 2 * fraction)
            displayed = (1 - blend) * initial + blend * goal
            with viewer.lock():
                robot.data.qpos[robot.qpos_indices] = displayed
                robot.data.qvel[:] = 0
                mujoco.mj_forward(robot.model, robot.data)
                _, current_position_error, current_rotation_error = pose_error(
                    robot.end_effector_transform(), target)
            phase = ("Moving to solution" if fraction < 1 else "Holding solution") if result.success else "Initial pose held (no solution)"
            labels = ["IK status", "Motion", "Position error (m)", "Rotation error (rad)"]
            values = [result.status, phase, f"{current_position_error:.5f}", f"{current_rotation_error:.5f}"]
            labels.extend(robot.joint_names)
            values.extend(f"{value:.4f} {'rad' if kind == 'revolute' else 'm'}"
                          for value, kind in zip(displayed, robot.joint_types))
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(labels), "\n".join(values)),
                              target_text])
            viewer.sync()
            remaining = 1 / 60 - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-viewer", action="store_true", help="Compute and print without opening a window")
    parser.add_argument("--position", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Override world position (m)")
    parser.add_argument("--rpy-deg", nargs=3, type=float, metavar=("ROLL", "PITCH", "YAW"), help="Override orientation (degrees)")
    parser.add_argument("--duration", type=float, default=3.0, help="Animation duration in seconds")
    parser.add_argument("--max-iterations", type=int, default=250, help="Iteration budget per starting guess")
    parser.add_argument("--restarts", type=int, default=8, help="Additional starting guesses")
    args = parser.parse_args(argv)
    try:
        if not np.isfinite(args.duration) or args.duration <= 0:
            raise ValueError("Animation duration must be positive and finite")
        target = end_effector_inputs.target_transform(args.position, args.rpy_deg)
        initial = (robot.home_q.copy() if end_effector_inputs.INITIAL_JOINTS is None
                   else validate_joint_angles(end_effector_inputs.INITIAL_JOINTS, robot.joint_count).copy())
        print(f"Model: {robot.xml_path}")
        print(f"End-effector body: {robot.end_effector_body_name}")
        print(f"Inputs: {end_effector_inputs.__file__}")
        print("Target pose in world coordinates:\n", np.array2string(target, precision=5, suppress_small=True))
        print("Solving IK...", flush=True)
        result = solve_target(target, initial, max_iterations=args.max_iterations, restarts=args.restarts)
    except ValueError as exc:
        print(f"INVALID_INPUT: {exc}")
        return 2
    print_result(result)
    if not args.no_viewer:
        print("Target axes: red x, green y, blue z. Close the viewer to exit.")
        target_rpy = end_effector_inputs.TARGET_RPY_DEG if args.rpy_deg is None else args.rpy_deg
        visualize_solution(target, initial, result, args.duration, target_rpy_deg=target_rpy)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
