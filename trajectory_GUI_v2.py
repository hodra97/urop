"""Run CLIK v2 with selectable references and a motion/reference clock.

ROBOT_MODULE defaults to panda_model; TRAJECTORY_MODULE defaults to
straight_line_trajectory. Nominal duration is 5 s, total motion budget 10 s.
Compute and verify first, then play the recorded kinematic states once and hold.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

import end_effector_inputs
from FK_algorithm import fk_manual, validate_joint_angles
from IK_algorithm import pose_error
import robot_config
import trajectory_config
from trajectory_solver_v2 import SolverSettings, sample_reference, solve_trajectory
from viewer_overlays import desired_pose_text, draw_path, draw_target_axes, draw_world_grid


def verify_result(robot, trajectory, result, settings):
    """Independently verify accepted states and interval probes with MuJoCo."""
    saved = robot.data.qpos.copy()
    actual, desired, errors, goal_errors, disagreements, guard_errors = [], [], [], [], [], []
    try:
        for i, (q, tau) in enumerate(zip(result.joints, result.taus)):
            target, _ = sample_reference(trajectory, tau)
            native = robot.transform_at(q)
            manual = fk_manual(q, robot.home_q, robot.home_axes, robot.home_anchors,
                               robot.home_transform, robot.joint_types)
            actual.append(native)
            desired.append(target)
            errors.append(pose_error(native, target)[1:])
            goal_errors.append(pose_error(native, trajectory.goal)[1:])
            disagreements.append(pose_error(native, manual)[1:])
            worst = np.array(errors[-1])
            if i:
                for fraction in (0.25, 0.5, 0.75):
                    probe_q = result.joints[i - 1] + fraction * (q - result.joints[i - 1])
                    probe_tau = result.taus[i - 1] + fraction * (tau - result.taus[i - 1])
                    target_probe, _ = sample_reference(trajectory, probe_tau)
                    worst = np.maximum(worst, pose_error(robot.transform_at(probe_q), target_probe)[1:])
            guard_errors.append(worst)
    finally:
        robot.data.qpos[:] = saved
        mujoco.mj_forward(robot.model, robot.data)
    actual, desired, errors, goal_errors, disagreements, guard_errors = map(
        np.asarray, (actual, desired, errors, goal_errors, disagreements, guard_errors))
    tolerance = np.array([settings.position_tolerance, settings.orientation_tolerance])
    checked_path_ok = bool(np.all(guard_errors <= tolerance * (1 + 1e-8)))
    fk_matches = bool(np.all(disagreements <= 1e-7))
    velocity, acceleration = result.joint_derivatives()
    lower, upper = robot.joint_limits()
    caps = np.array([settings.revolute_speed if kind == "revolute" else settings.prismatic_speed
                     for kind in robot.joint_types])
    joints_ok = bool(np.all(result.joints >= lower - 1e-12) and np.all(result.joints <= upper + 1e-12)
                     and np.all(np.abs(velocity) <= caps * (1 + 1e-8)))
    completed = bool(result.taus[-1] >= trajectory.duration - 1e-12)
    if (not checked_path_ok or not fk_matches or not joints_ok
            or (result.success and (not completed or np.any(goal_errors[-1] > tolerance * (1 + 1e-8))))):
        result.status, result.message = "VERIFICATION_FAILED", "MuJoCo or joint-limit verification rejected the numerical result"
    metrics = dict(
        status=result.status, message=result.message, completed=completed,
        motion_time_s=float(result.times[-1]), reference_time_s=float(result.taus[-1]),
        nominal_duration_s=float(trajectory.duration), max_time_s=settings.max_time,
        nominal_time_met=bool(result.success and result.times[-1] <= trajectory.duration + 1e-9),
        samples=len(result.times), phase_limited_intervals=int(np.sum(result.alphas[1:] < 1 - 1e-9)),
        velocity_bound_intervals=int(result.velocity_bounded.sum()), rejected_candidates=int(result.rejected_candidates.sum()),
        min_alpha=float(np.min(result.alphas[1:])) if len(result.alphas) > 1 else 0.0,
        max_position_error_at_tau_m=float(np.max(errors[:, 0])),
        max_orientation_error_at_tau_rad=float(np.max(errors[:, 1])),
        rms_position_error_at_tau_m=float(np.sqrt(np.mean(errors[:, 0]**2))),
        rms_orientation_error_at_tau_rad=float(np.sqrt(np.mean(errors[:, 1]**2))),
        max_checked_position_error_m=float(np.max(guard_errors[:, 0])),
        max_checked_orientation_error_rad=float(np.max(guard_errors[:, 1])),
        checked_path_within_tolerance=checked_path_ok,
        final_goal_position_error_m=float(goal_errors[-1, 0]),
        final_goal_orientation_error_rad=float(goal_errors[-1, 1]),
        max_abs_joint_velocity=np.max(np.abs(velocity), axis=0).tolist(),
        max_abs_joint_acceleration=np.max(np.abs(acceleration), axis=0).tolist(),
        joint_limits_verified=joints_ok, mujoco_verified=fk_matches,
        max_fk_position_disagreement_m=float(np.max(disagreements[:, 0])),
        max_fk_orientation_disagreement_rad=float(np.max(disagreements[:, 1])),
        computation_seconds=result.computation_seconds,
    )
    return dict(actual=actual, desired=desired, errors=errors, goal_errors=goal_errors,
                guard_errors=guard_errors, velocity=velocity, acceleration=acceleration, metrics=metrics)


def export_result(directory, robot, trajectory, result, measured, settings, *, trajectory_module):
    """Export v2 clocks and validated poses; never overwrite an experiment."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    pose_columns = ["x_m", "y_m", "z_m"] + [f"R{i}{j}" for i in range(3) for j in range(3)]

    def pose_values(pose):
        return np.r_[pose[:3, 3], pose[:3, :3].ravel()]

    with (directory / "clik.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "tau_s", "reference_time_fraction", "alpha", "position_error_at_tau_m",
                         "orientation_error_at_tau_rad", "goal_position_error_m", "goal_orientation_error_rad",
                         "checked_position_error_m", "checked_orientation_error_rad", "velocity_bounded",
                         "rejected_candidates", "step_seconds",
                         *[f"desired_{c}" for c in pose_columns], *[f"actual_{c}" for c in pose_columns],
                         *[f"q_{j}" for j in robot.joint_names], *[f"qdot_{j}" for j in robot.joint_names],
                         *[f"qddot_{j}" for j in robot.joint_names]])
        for i, t in enumerate(result.times):
            writer.writerow([t, result.taus[i], result.taus[i] / trajectory.duration, result.alphas[i],
                             *measured["errors"][i], *measured["goal_errors"][i], *measured["guard_errors"][i],
                             int(result.velocity_bounded[i]), result.rejected_candidates[i], result.step_seconds[i],
                             *pose_values(measured["desired"][i]), *pose_values(measured["actual"][i]),
                             *result.joints[i], *measured["velocity"][i], *measured["acceleration"][i]])
    with (directory / "reference.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tau_s", *pose_columns, "wx_rad_s", "wy_rad_s", "wz_rad_s", "vx_m_s", "vy_m_s", "vz_m_s"])
        for tau in np.r_[np.arange(0, trajectory.duration, settings.dt), trajectory.duration]:
            pose, twist = sample_reference(trajectory, tau)
            writer.writerow([tau, *pose_values(pose), *twist])
    lower, upper = robot.joint_limits()
    pin = Path(__file__).with_name("MENAGERIE_COMMIT.txt")
    report = dict(
        version=2, method="CLIK", created_utc=datetime.now(timezone.utc).isoformat(),
        model=str(robot.xml_path), end_effector_body=robot.end_effector_body_name,
        trajectory_module=trajectory_module, joint_names=list(robot.joint_names), joint_types=list(robot.joint_types),
        lower_limits=[float(v) if np.isfinite(v) else None for v in lower],
        upper_limits=[float(v) if np.isfinite(v) else None for v in upper],
        initial_joints=result.joints[0].tolist(), start=np.asarray(trajectory.start).tolist(),
        goal=np.asarray(trajectory.goal).tolist(), settings=asdict(settings),
        numpy_version=np.__version__, mujoco_version=mujoco.__version__,
        menagerie_reference_commit=pin.read_text().strip() if pin.exists() else None,
        results=measured["metrics"],
        notes=["t is motion/playback time; tau is nominal reference time. Neither is solver computation time.",
               "alpha describes the interval ending at each row; row zero uses alpha=0.",
               "reference_time_fraction is tau/duration, not geometric path progress s.",
               "Errors are relative to T_d(tau), including checks at 1/4, 1/2, 3/4 of each joint segment.",
               "Finite probes are not a continuous-time, collision, or physical actuator guarantee.",
               "Acceleration is reported but not constrained. Joint units are rad or m according to joint_types.",
               "The Menagerie commit is recorded as a reference, not an automatic checkout verification."],
    )
    (directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return directory


def visualize_trajectory(robot, trajectory, result, measured):
    """Replay accepted samples once and hold the final state, including failures."""
    reference = np.array([sample_reference(trajectory, tau)[0][:3, 3]
                          for tau in np.linspace(0, trajectory.duration, 150)])
    robot.reset()
    robot.transform_at(result.joints[0])
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        started = time.perf_counter()
        while viewer.is_running():
            frame = time.perf_counter()
            playback_time = min(max(0, frame - started), result.times[-1])
            index = max(0, min(int(np.searchsorted(result.times, playback_time, side="right")) - 1,
                               len(result.times) - 1))
            if result.status == "VERIFICATION_FAILED":
                index = 0
            tau = result.taus[index]
            target, _ = sample_reference(trajectory, tau)
            with viewer.lock():
                robot.data.qvel[:] = 0
                current = robot.transform_at(result.joints[index])
                viewer.user_scn.ngeom = 0
                draw_world_grid(viewer.user_scn)
                draw_target_axes(viewer.user_scn, target)
                draw_path(viewer.user_scn, reference, [1, 0.8, 0.1, 1])
                if result.status != "VERIFICATION_FAILED":
                    draw_path(viewer.user_scn, measured["actual"][:, :3, 3], [0.2, 1, 0.3, 1])
            _, position, rotation = pose_error(current, target)
            labels = ("Method\nResult\nMotion time t (s)\nReference time tau (s)\nNominal duration (s)"
                      "\nAlpha\nPosition error at tau (m)\nRotation error at tau (rad)\nPlayback\nPaths")
            phase = "Holding last recorded pose" if playback_time >= result.times[-1] else "Playing computed trajectory"
            values = (f"CLIK v2\n{result.status}\n{result.times[index]:.3f}\n{tau:.3f}\n{trajectory.duration:.3f}"
                      f"\n{result.alphas[index]:.4f}\n{position:.6g}\n{rotation:.6g}\n{phase}\nDesired yellow / Actual green")
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               labels, values), desired_pose_text(target)])
            viewer.sync()
            remaining = 1 / 60 - (time.perf_counter() - frame)
            if remaining > 0:
                time.sleep(remaining)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--trajectory-module", help="Override TRAJECTORY_MODULE for this run")
    position = parser.add_mutually_exclusive_group()
    position.add_argument("--position", nargs=3, type=float, help="Absolute world XYZ (m)")
    position.add_argument("--offset", nargs=3, type=float, help="World XYZ displacement from initial pose (m)")
    rotation = parser.add_mutually_exclusive_group()
    rotation.add_argument("--rpy-deg", nargs=3, type=float, help="Absolute world RPY (degrees)")
    rotation.add_argument("--rotation-offset-deg", nargs=3, type=float, help="Local increment R_goal = R_start @ Rz Ry Rx")
    parser.add_argument("--duration", type=float, default=5.0, help="Nominal reference duration, seconds")
    defaults = SolverSettings()
    for name, value in asdict(defaults).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int if name == "phase_search_iterations" else float,
                            default=value)
    parser.add_argument("--output", type=Path, help="New output directory; default: unique runs/trajectory_v2_TIMESTAMP")
    args = parser.parse_args(argv)
    try:
        settings = SolverSettings(**{name: getattr(args, name) for name in asdict(defaults)})
        if not np.isfinite(args.duration) or args.duration <= 0:
            raise ValueError("duration must be positive and finite")
        output = args.output or Path("runs") / datetime.now().strftime("trajectory_v2_%Y%m%d_%H%M%S_%f")
        if output.exists():
            raise ValueError(f"Output directory already exists: {output}")
        robot = robot_config.load_robot()
        initial = (robot.home_q.copy() if end_effector_inputs.INITIAL_JOINTS is None
                   else validate_joint_angles(end_effector_inputs.INITIAL_JOINTS, robot.joint_count).copy())
        start = robot.transform_at(initial)
        goal = end_effector_inputs.target_transform(args.position, args.rpy_deg)
        if args.offset is not None:
            goal[:3, 3] = start[:3, 3] + np.asarray(args.offset)
        if args.rotation_offset_deg is not None:
            goal[:3, :3] = start[:3, :3] @ end_effector_inputs.target_transform([0, 0, 0], args.rotation_offset_deg)[:3, :3]
        profile = trajectory_config.load_profile(args.trajectory_module)
        trajectory = trajectory_config.load_trajectory(start, goal, args.duration, profile)
        lower, upper = robot.joint_limits()
        print(f"Model: {robot.xml_path}\nEnd-effector body: {robot.end_effector_body_name}")
        print(f"Trajectory: {profile.__name__}; nominal {trajectory.duration:g}s; total budget {settings.max_time:g}s")
        print("Computing CLIK v2 with adaptive reference time...", flush=True)
        result = solve_trajectory(trajectory, initial, robot.home_q, robot.home_axes, robot.home_anchors,
                                  robot.home_transform, robot.joint_types, lower_limits=lower, upper_limits=upper,
                                  settings=settings)
        print("Verifying recorded states and intermediate probes with MuJoCo...", flush=True)
        measured = verify_result(robot, trajectory, result, settings)
        output = export_result(output, robot, trajectory, result, measured, settings, trajectory_module=profile.__name__)
    except (ValueError, OSError, ImportError, AttributeError, TypeError) as exc:
        print(f"INVALID_INPUT: {exc}")
        return 2
    metrics = measured["metrics"]
    print(f"\n{result.status}: {result.message}")
    print(f"Motion time t: {result.times[-1]:.3f}s; reference time tau: {result.taus[-1]:.3f}/{trajectory.duration:g}s")
    print(f"Final goal error: {metrics['final_goal_position_error_m']:.6g} m / {metrics['final_goal_orientation_error_rad']:.6g} rad")
    print(f"Max checked reference error: {metrics['max_checked_position_error_m']:.6g} m / {metrics['max_checked_orientation_error_rad']:.6g} rad")
    print(f"Phase-limited intervals: {metrics['phase_limited_intervals']}; minimum alpha: {metrics['min_alpha']:.4f}")
    print(f"Checked path within tolerance: {metrics['checked_path_within_tolerance']}; MuJoCo verified: {metrics['mujoco_verified']}")
    print(f"Computation: {result.computation_seconds:.3f}s; records: {output.resolve()}")
    if not args.no_viewer:
        visualize_trajectory(robot, trajectory, result, measured)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
