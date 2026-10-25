"""Compute BOTH pointwise DLS IK and CLIK on the same pose/time reference.

Pure numerical code: geometry is passed explicitly. Neither solver consumes
the other one's result. Both start at the supplied q_initial. CLIK uses a
weighted DLS velocity inverse, feedforward plus proportional pose feedback,
and explicit Euler integration; it does not iterate static IK at each step.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from FK_algorithm import fk_manual, validate_joint_angles, validate_joint_types
from IK_algorithm import pose_error, solve_ik
from jacobian_algorithm import adjoint, jacobian_space, transform_inverse


@dataclass(frozen=True)
class SolverSettings:
    dt: float = 0.01
    settle_time: float = 1.0
    gain: float = 5.0                 # proportional CLIK gain, 1/s
    damping: float = 1e-3
    orientation_scale: float = 0.3   # m/rad, same weighting as static IK
    position_tolerance: float = 1e-4
    orientation_tolerance: float = 1e-3
    max_iterations: int = 250        # per DLS IK waypoint; restarts are disabled
    revolute_speed: float = 1.0      # rad/s; experimental cap, not actuator rating
    prismatic_speed: float = 0.1     # m/s; experimental cap, not actuator rating

    def __post_init__(self):
        for name in ("dt", "gain", "damping", "orientation_scale", "position_tolerance",
                     "orientation_tolerance", "revolute_speed", "prismatic_speed"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(self.settle_time) or self.settle_time < 0:
            raise ValueError("settle_time must be nonnegative and finite")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")


@dataclass
class TrajectoryResult:
    method: str
    times: np.ndarray
    joints: np.ndarray
    iterations: np.ndarray
    step_seconds: np.ndarray
    speed_limited: np.ndarray
    joint_limited: np.ndarray
    status: str
    message: str
    computation_seconds: float
    failed_time: float | None = None

    @property
    def success(self):
        return self.status == "SUCCESS"

    def joint_derivatives(self):
        """Backward interval differences, with zero initial velocity/acceleration.

These are sample-based diagnostics, not actuator commands or acceleration limits.
"""
        velocity = np.zeros_like(self.joints)
        acceleration = np.zeros_like(self.joints)
        if len(self.times) > 1:
            intervals = np.diff(self.times)[:, None]
            velocity[1:] = np.diff(self.joints, axis=0) / intervals
            acceleration[1:] = np.diff(velocity, axis=0) / intervals
        return velocity, acceleration


def solve_trajectories(trajectory, q_initial, home_q, home_axes, home_anchors,
                       home_transform, joint_types=None, *, lower_limits=None,
                       upper_limits=None, settings=None):
    """Return {'DLS_IK': result, 'CLIK': result}, even if one method fails.

DLS IK warm-starts each waypoint from its own previous solution, with no random
restarts. An infeasible waypoint or speed jump stops that result at its last
accepted sample. CLIK scales velocity to the same caps and projects onto joint
limits; its actual tracking errors must be measured separately. SUCCESS means
the final pose is within tolerance, not that every intermediate sample is.
"""
    settings = SolverSettings() if settings is None else settings
    home_q = validate_joint_angles(home_q).copy()
    initial = validate_joint_angles(q_initial, len(home_q)).copy()
    kinds = validate_joint_types(joint_types, len(home_q))
    lower = np.full_like(initial, -np.inf) if lower_limits is None else np.asarray(lower_limits, dtype=float).copy()
    upper = np.full_like(initial, np.inf) if upper_limits is None else np.asarray(upper_limits, dtype=float).copy()
    geometry = (home_q, home_axes, home_anchors, home_transform, kinds)
    options = dict(lower_limits=lower, upper_limits=upper,
                   position_tolerance=settings.position_tolerance,
                   orientation_tolerance=settings.orientation_tolerance,
                   max_iterations=settings.max_iterations, restarts=0,
                   damping=settings.damping, orientation_scale=settings.orientation_scale)
    initial_pose = fk_manual(initial, *geometry)
    # Reuse the existing solver's validation without moving the initial posture.
    solve_ik(initial_pose, initial, *geometry, **options)
    _, start_position, start_rotation = pose_error(initial_pose, trajectory.start)
    if start_position > 1e-8 or start_rotation > 1e-8:
        raise ValueError("Trajectory start must match FK(q_initial)")
    times = trajectory.sample_times(settings.dt, settings.settle_time)
    references = [trajectory.sample(t) for t in times]
    caps = np.array([settings.revolute_speed if kind == "revolute" else settings.prismatic_speed
                     for kind in kinds])
    results = {}
    for method in ("DLS_IK", "CLIK"):
        started = perf_counter()
        q = initial.copy()
        joints, iterations, costs = [q.copy()], [0], [0.0]
        speed_limited, joint_limited = [False], [False]
        status, message, failed_time = "SUCCESS", "Final pose reached", None
        for index in range(1, len(times)):
            step_started = perf_counter()
            interval = times[index] - times[index - 1]
            if method == "DLS_IK":
                solved = solve_ik(references[index][0], q, *geometry, **options)
                if not solved.success:
                    status, message = solved.status, solved.message
                    failed_time = float(times[index])
                    break
                if np.any(np.abs(solved.q - q) / interval > caps * (1 + 1e-9)):
                    status = "SPEED_LIMIT"
                    message = "DLS waypoint exceeds a joint speed cap; increase duration or revise the target"
                    failed_time = float(times[index])
                    break
                q = solved.q.copy()
                count, speed_hit, joint_hit = solved.iterations, False, False
            else:
                current = fk_manual(q, *geometry)
                desired, desired_twist = references[index - 1]
                error, _, _ = pose_error(current, desired)
                relative = transform_inverse(current) @ desired
                commanded = adjoint(relative) @ desired_twist + settings.gain * error
                body = adjoint(transform_inverse(current)) @ jacobian_space(
                    q, home_q, home_axes, home_anchors, kinds)
                weights = np.array([settings.orientation_scale] * 3 + [1.0] * 3)
                weighted = weights[:, None] * body
                residual = weights * commanded
                gradient = weighted.T @ residual
                free = ~(((q <= lower + 1e-10) & (gradient < 0))
                         | ((q >= upper - 1e-10) & (gradient > 0)) | (lower == upper))
                active = weighted[:, free]
                velocity = np.zeros_like(q)
                velocity[free] = active.T @ np.linalg.solve(
                    active @ active.T + settings.damping**2 * np.eye(6), residual)
                scale = max(1.0, float(np.max(np.abs(velocity) / caps)))
                speed_hit = scale > 1.0
                candidate = q + interval * velocity / scale
                joint_hit = bool(np.any(~free) or np.any(candidate < lower) or np.any(candidate > upper))
                q = np.clip(candidate, lower, upper)
                count = 1
            joints.append(q.copy())
            iterations.append(count)
            costs.append(perf_counter() - step_started)
            speed_limited.append(speed_hit)
            joint_limited.append(joint_hit)
        if status == "SUCCESS":
            _, position_error, orientation_error = pose_error(fk_manual(q, *geometry), references[-1][0])
            if position_error > settings.position_tolerance or orientation_error > settings.orientation_tolerance:
                status, message = "NO_CONVERGENCE", "Final pose outside tolerance after the allotted tracking/hold time"
        results[method] = TrajectoryResult(
            method, times[:len(joints)].copy(), np.array(joints), np.array(iterations),
            np.array(costs), np.array(speed_limited), np.array(joint_limited),
            status, message, perf_counter() - started, failed_time)
    return results
