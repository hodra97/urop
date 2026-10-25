"""Compare DLS IK and CLIK, verify with MuJoCo, export, and replay both.

The viewer plays computed joint samples using mj_forward, without actuator
tracking or collision avoidance. Both methods are always computed.
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
from straight_line_trajectory import StraightLineTrajectory
from trajectory_solver_v1 import SolverSettings, solve_trajectories
from viewer_overlays import desired_pose_text, draw_target_axes, draw_world_grid


def verify_results(robot, trajectory, results, settings):
    """Measure every accepted joint sample independently with MuJoCo.

Restore the model's qpos afterwards so verification does not choose a playback
start or leak the final posture into a subsequent experiment.
"""
    measurements = {}
    saved = robot.data.qpos.copy()
    try:
        for name, result in results.items():
            actual, desired, errors, discrepancies, goal_errors = [], [], [], [], []
            for t, q in zip(result.times, result.joints):
                target, _ = trajectory.sample(t)
                native = robot.transform_at(q)
                manual = fk_manual(q, robot.home_q, robot.home_axes, robot.home_anchors,
                                   robot.home_transform, robot.joint_types)
                desired.append(target)
                actual.append(native)
                errors.append(pose_error(native, target)[1:])
                discrepancies.append(pose_error(native, manual)[1:])
                goal_errors.append(pose_error(native, trajectory.goal)[1:])
            errors, discrepancies, goal_errors = map(np.asarray, (errors, discrepancies, goal_errors))
            within = ((errors[:, 0] <= settings.position_tolerance)
                      & (errors[:, 1] <= settings.orientation_tolerance))
            verified = bool(np.all(discrepancies <= 1e-7))
            if not verified or (result.success and (not within[-1] or
                                                    (name == "DLS_IK" and not np.all(within)))):
                result.status, result.message = "VERIFICATION_FAILED", "MuJoCo disagrees with the numerical result"
            at_T = np.flatnonzero(np.isclose(result.times, trajectory.duration, rtol=0, atol=1e-12))
            complete = bool(np.isclose(result.times[-1], trajectory.duration + settings.settle_time,
                                       rtol=0, atol=1e-12))
            goal_within = ((goal_errors[:, 0] <= settings.position_tolerance)
                           & (goal_errors[:, 1] <= settings.orientation_tolerance))
            stayed = np.flatnonzero(np.logical_and.accumulate(goal_within[::-1])[::-1])
            velocity, acceleration = result.joint_derivatives()
            metrics = dict(
                status=result.status, message=result.message, completed=complete,
                samples=len(result.times), failed_time=result.failed_time,
                max_position_error_m=float(np.max(errors[:, 0])),
                rms_position_error_m=float(np.sqrt(np.mean(errors[:, 0]**2))),
                max_orientation_error_rad=float(np.max(errors[:, 1])),
                rms_orientation_error_rad=float(np.sqrt(np.mean(errors[:, 1]**2))),
                position_error_at_T_m=float(errors[at_T[0], 0]) if len(at_T) else None,
                orientation_error_at_T_rad=float(errors[at_T[0], 1]) if len(at_T) else None,
                last_position_error_m=float(errors[-1, 0]), last_orientation_error_rad=float(errors[-1, 1]),
                path_within_tolerance=bool(complete and np.all(within)),
                goal_enter_and_stay_time_s=float(result.times[stayed[0]]) if complete and len(stayed) else None,
                speed_limit_events=int(result.speed_limited.sum()) + int(result.status == "SPEED_LIMIT"),
                joint_limit_events=int(result.joint_limited.sum()),
                max_abs_joint_velocity=np.max(np.abs(velocity), axis=0).tolist(),
                max_abs_joint_acceleration=np.max(np.abs(acceleration), axis=0).tolist(),
                computation_seconds=result.computation_seconds,
                mujoco_verified=verified,
                max_fk_position_disagreement_m=float(np.max(discrepancies[:, 0])),
                max_fk_orientation_disagreement_rad=float(np.max(discrepancies[:, 1])),
            )
            measurements[name] = dict(actual=np.array(actual), desired=np.array(desired),
                                      errors=errors, velocity=velocity, acceleration=acceleration,
                                      metrics=metrics)
    finally:
        robot.data.qpos[:] = saved
        mujoco.mj_forward(robot.model, robot.data)
    return measurements


def export_results(directory, robot, trajectory, initial, results, measurements, settings):
    """Save complete reference, native poses, joint diagnostics and run settings.

An existing directory is rejected to keep previous experiment records intact.
Failed DLS paths remain truncated; reference.csv still contains the whole goal.
"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    pose_columns = ["x_m", "y_m", "z_m"] + [f"R{i}{j}" for i in range(3) for j in range(3)]

    def pose_values(pose):
        return np.r_[pose[:3, 3], pose[:3, :3].ravel()]

    with (directory / "reference.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", *pose_columns, "wx_rad_s", "wy_rad_s", "wz_rad_s", "vx_m_s", "vy_m_s", "vz_m_s"])
        for t in trajectory.sample_times(settings.dt, settings.settle_time):
            pose, twist = trajectory.sample(t)
            writer.writerow([t, *pose_values(pose), *twist])
    for name, result in results.items():
        measured = measurements[name]
        with (directory / f"{name.lower()}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time_s", "position_error_m", "orientation_error_rad", "iterations",
                             "step_seconds", "speed_limited", "joint_limited",
                             *[f"desired_{c}" for c in pose_columns], *[f"actual_{c}" for c in pose_columns],
                             *[f"q_{j}" for j in robot.joint_names],
                             *[f"qdot_{j}" for j in robot.joint_names],
                             *[f"qddot_{j}" for j in robot.joint_names]])
            for i, t in enumerate(result.times):
                writer.writerow([t, *measured["errors"][i], result.iterations[i], result.step_seconds[i],
                                 int(result.speed_limited[i]), int(result.joint_limited[i]),
                                 *pose_values(measured["desired"][i]), *pose_values(measured["actual"][i]),
                                 *result.joints[i], *measured["velocity"][i], *measured["acceleration"][i]])
    lower, upper = robot.joint_limits()
    def finite_limits(values):
        return [float(v) if np.isfinite(v) else None for v in values]

    pin = Path(__file__).with_name("MENAGERIE_COMMIT.txt")
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), model=str(robot.xml_path),
        end_effector_body=robot.end_effector_body_name, joint_names=list(robot.joint_names),
        joint_types=list(robot.joint_types), lower_limits=finite_limits(lower), upper_limits=finite_limits(upper),
        unlimited_limit_encoding=None, initial_joints=initial.tolist(), duration=trajectory.duration,
        start=trajectory.start.tolist(), goal=trajectory.goal.tolist(), settings=asdict(settings),
        numpy_version=np.__version__, mujoco_version=mujoco.__version__,
        menagerie_reference_commit=pin.read_text().strip() if pin.exists() else None,
        notes=["The reference commit is recorded, not an automatic checkout verification.",
               "Kinematic samples only; no collision, actuator or acceleration-limit verification.",
               "Joint units are rad for revolute joints and m for prismatic joints.",
               "Reference twists use desired-body coordinates, angular first.",
               "RMS is an unweighted sample RMS; derivatives are backward interval differences.",
               "Solver timing excludes model loading, MuJoCo verification, export and playback.",
               "SUCCESS checks the final pose; path_within_tolerance checks every recorded reference sample."],
        results={name: measured["metrics"] for name, measured in measurements.items()},
    )
    (directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return directory


def _draw_path(scene, points, color):
    # Limit display geometry only; exported data keeps every sample.
    points = points[np.unique(np.linspace(0, len(points) - 1, min(150, len(points))).astype(int))]
    for start, end in zip(points[:-1], points[1:]):
        if scene.ngeom >= scene.maxgeom:
            break
        if np.linalg.norm(end - start) < 1e-12:
            continue
        geom = scene.geoms[scene.ngeom]
        kind = mujoco.mjtGeom.mjGEOM_LINE
        mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color, dtype=np.float32))
        mujoco.mjv_connector(geom, kind, 3.0, start, end)
        scene.ngeom += 1


def visualize_trajectories(robot, trajectory, results, measurements, settings):
    """Alternate DLS IK and CLIK playback, resetting to the same start each time.

Play recorded samples (sample-and-hold), not an initial-to-final joint blend.
If a solve stopped early, hold its last accepted sample and display failure.
"""
    names = tuple(results)
    length = trajectory.duration + settings.settle_time + 1.0
    colors = {"DLS_IK": [0.1, 0.8, 1.0, 1.0], "CLIK": [0.3, 1.0, 0.3, 1.0]}
    robot.reset()
    robot.transform_at(results[names[0]].joints[0])
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        started = time.perf_counter()
        while viewer.is_running():
            frame = time.perf_counter()
            elapsed = max(0.0, frame - started)
            name = names[int(elapsed / length) % len(names)]
            result = results[name]
            t = min(elapsed % length, trajectory.duration + settings.settle_time)
            index = min(int(np.searchsorted(result.times, t, side="right")) - 1, len(result.times) - 1)
            index = max(0, index)
            target, _ = trajectory.sample(t)
            with viewer.lock():
                robot.data.qvel[:] = 0
                current = robot.transform_at(result.joints[index])
                viewer.user_scn.ngeom = 0
                draw_world_grid(viewer.user_scn)
                draw_target_axes(viewer.user_scn, target)
                _draw_path(viewer.user_scn, np.array([trajectory.start[:3, 3], trajectory.goal[:3, 3]]), [1, 0.8, 0.1, 1])
                for method in names:
                    _draw_path(viewer.user_scn, measurements[method]["actual"][:, :3, 3], colors[method])
            _, pos, rot = pose_error(current, target)
            labels = "Method\nStatus\nReference time (s)\nRecorded sample (s)\nPosition error (m)\nRotation error (rad)\nPaths\nPlayback"
            values = (f"{name}\n{result.status}\n{t:.3f}\n{result.times[index]:.3f}\n{pos:.6g}\n{rot:.6g}"
                      "\nDesired yellow / DLS cyan / CLIK green\nBoth methods alternate; close to exit")
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               labels, values), desired_pose_text(target)])
            viewer.sync()
            remaining = 1 / 60 - (time.perf_counter() - frame)
            if remaining > 0:
                time.sleep(remaining)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-viewer", action="store_true")
    position = parser.add_mutually_exclusive_group()
    position.add_argument("--position", nargs=3, type=float, help="Absolute world XYZ (m); default: end_effector_inputs")
    position.add_argument("--offset", nargs=3, type=float, help="World XYZ displacement from initial pose (m)")
    rotation = parser.add_mutually_exclusive_group()
    rotation.add_argument("--rpy-deg", nargs=3, type=float, help="Absolute world RPY (degrees); default: end_effector_inputs")
    rotation.add_argument("--rotation-offset-deg", nargs=3, type=float, help="Local RPY increment: R_goal = R_start @ Rz Ry Rx")
    parser.add_argument("--duration", type=float, default=5.0, help="Reference travel time, seconds")
    defaults = SolverSettings()
    for name in ("dt", "settle_time", "gain", "damping", "orientation_scale",
                 "position_tolerance", "orientation_tolerance", "revolute_speed", "prismatic_speed"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--max-iterations", type=int, default=defaults.max_iterations)
    parser.add_argument("--output", type=Path, help="New output directory; default: unique runs/trajectory_TIMESTAMP")
    args = parser.parse_args(argv)
    try:
        settings = SolverSettings(**{name: getattr(args, name) for name in asdict(defaults)})
        output = args.output or Path("runs") / datetime.now().strftime("trajectory_%Y%m%d_%H%M%S_%f")
        if output.exists():
            raise ValueError(f"Output directory already exists: {output}; choose a new path")
        robot = robot_config.load_robot()
        initial = (robot.home_q.copy() if end_effector_inputs.INITIAL_JOINTS is None
                   else validate_joint_angles(end_effector_inputs.INITIAL_JOINTS, robot.joint_count).copy())
        start = robot.transform_at(initial)
        goal = end_effector_inputs.target_transform(args.position, args.rpy_deg)
        if args.offset is not None:
            goal[:3, 3] = start[:3, 3] + np.asarray(args.offset)
        if args.rotation_offset_deg is not None:
            goal[:3, :3] = start[:3, :3] @ end_effector_inputs.target_transform([0, 0, 0], args.rotation_offset_deg)[:3, :3]
        trajectory = StraightLineTrajectory(start, goal, args.duration)
        lower, upper = robot.joint_limits()
        print(f"Model: {robot.xml_path}\nEnd-effector body: {robot.end_effector_body_name}")
        print("Computing BOTH DLS IK and CLIK from the same initial posture...", flush=True)
        results = solve_trajectories(
            trajectory, initial, robot.home_q, robot.home_axes, robot.home_anchors,
            robot.home_transform, robot.joint_types, lower_limits=lower, upper_limits=upper, settings=settings)
        measured = verify_results(robot, trajectory, results, settings)
        output = export_results(output, robot, trajectory, initial, results, measured, settings)
    except (ValueError, OSError, ImportError) as exc:
        print(f"INVALID_INPUT: {exc}")
        return 2
    for name, result in results.items():
        metrics = measured[name]["metrics"]
        print(f"\n{name}: {result.status} - {result.message}")
        print(f"  Recorded until {result.times[-1]:.3f}s; samples: {len(result.times)}; compute: {result.computation_seconds:.3f}s")
        print(f"  Max / RMS position error: {metrics['max_position_error_m']:.6g} / {metrics['rms_position_error_m']:.6g} m")
        print(f"  Max / RMS orientation error: {metrics['max_orientation_error_rad']:.6g} / {metrics['rms_orientation_error_rad']:.6g} rad")
        print(f"  At T: position={metrics['position_error_at_T_m']}, orientation={metrics['orientation_error_at_T_rad']}")
        print(f"  Last recorded sample: position={metrics['last_position_error_m']:.6g} m, orientation={metrics['last_orientation_error_rad']:.6g} rad")
        print(f"  Speed / joint-limit events: {metrics['speed_limit_events']} / {metrics['joint_limit_events']}")
        print(f"  Entire sampled path within tolerance: {metrics['path_within_tolerance']}; MuJoCo verified: {metrics['mujoco_verified']}")
        if result.failed_time is not None:
            print(f"  Failed waypoint time: {result.failed_time:.6g}s")
    print(f"\nCSV and JSON results: {output.resolve()}")
    if not args.no_viewer:
        visualize_trajectories(robot, trajectory, results, measured, settings)
    return 0 if all(result.success for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
