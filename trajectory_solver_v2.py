"""CLIK with a virtual reference clock, bounded DLS, and checked pose errors.

No per-waypoint static IK is called. A reference provides start, goal, duration,
and sample(tau) -> (world pose, nominal desired-body twist [omega; v]).
The motion clock t advances by dt; the reference clock advances by alpha * dt.
The solver receives numerical geometry and never loads a robot or a profile.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from FK_algorithm import fk_manual, validate_geometry, validate_joint_angles, validate_joint_types
from IK_algorithm import pose_error, rotation_log
from jacobian_algorithm import adjoint, jacobian_space, transform_inverse


@dataclass(frozen=True)
class SolverSettings:
    dt: float = 0.01
    max_time: float = 10.0            # Total motion time, not extra time after T
    stall_time: float = 1.0           # No reference progress or error improvement
    gain: float = 5.0
    damping: float = 1e-3
    orientation_scale: float = 0.3
    position_tolerance: float = 1e-4  # Error relative to the pose at tau
    orientation_tolerance: float = 1e-3
    revolute_speed: float = 1.0      # Experimental cap, not a hardware rating
    prismatic_speed: float = 0.1
    phase_search_iterations: int = 10

    def __post_init__(self):
        for name in ("dt", "max_time", "stall_time", "gain", "damping", "orientation_scale",
                     "position_tolerance", "orientation_tolerance", "revolute_speed", "prismatic_speed"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.phase_search_iterations, int) or not 1 <= self.phase_search_iterations <= 30:
            raise ValueError("phase_search_iterations must be an integer from 1 to 30")


@dataclass
class TrajectoryResult:
    times: np.ndarray
    taus: np.ndarray
    alphas: np.ndarray              # Interval ending at this row; row 0 is zero
    joints: np.ndarray
    guard_errors: np.ndarray        # Maximum over endpoint + 3 interior probes
    velocity_bounded: np.ndarray
    rejected_candidates: np.ndarray
    step_seconds: np.ndarray
    status: str
    message: str
    computation_seconds: float

    @property
    def success(self):
        return self.status == "SUCCESS"

    def joint_derivatives(self):
        velocity, acceleration = np.zeros_like(self.joints), np.zeros_like(self.joints)
        if len(self.times) > 1:
            intervals = np.diff(self.times)[:, None]
            velocity[1:] = np.diff(self.joints, axis=0) / intervals
            acceleration[1:] = np.diff(velocity, axis=0) / intervals
        return velocity, acceleration


def sample_reference(trajectory, tau):
    """Validate every reference sample, including when called without a loader."""
    pose, twist = trajectory.sample(float(tau))
    pose, twist = np.array(pose, dtype=float, copy=True), np.array(twist, dtype=float, copy=True)
    transform_inverse(pose)
    if twist.shape != (6,) or not np.all(np.isfinite(twist)):
        raise ValueError("Trajectory twist must be a finite six-vector [omega; v]")
    return pose, twist


def bounded_dls(jacobian, command, lower, upper, damping):
    """Box-constrained damped least squares using a small active-set QP.

Minimize ||J v - command||^2 + damping^2 ||v||^2 with lower <= v <= upper.
Unlike clipping an unconstrained answer, free joints are solved again when a
bound becomes active. This lets redundant arms use their remaining joints.
"""
    count = jacobian.shape[1]
    hessian = jacobian.T @ jacobian + damping**2 * np.eye(count)
    rhs = jacobian.T @ command
    velocity = np.clip(np.zeros(count), lower, upper)
    fixed = upper - lower <= 1e-12
    active = np.zeros(count, dtype=int)
    active[fixed] = -1
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(rhs))))
    for _ in range(20 * count + 20):
        free = active == 0
        candidate = velocity.copy()
        if np.any(free):
            candidate[free] = np.linalg.solve(
                hessian[np.ix_(free, free)], rhs[free] - hessian[np.ix_(free, ~free)] @ velocity[~free])
        direction = candidate - velocity
        fractions = np.full(count, np.inf)
        below = free & (candidate < lower - 1e-12)
        above = free & (candidate > upper + 1e-12)
        fractions[below] = (lower[below] - velocity[below]) / direction[below]
        fractions[above] = (upper[above] - velocity[above]) / direction[above]
        if np.any(below | above):
            blocking = int(np.argmin(fractions))
            fraction = float(np.clip(fractions[blocking], 0.0, 1.0))
            velocity = np.clip(velocity + fraction * direction, lower, upper)
            active[blocking] = -1 if below[blocking] else 1
            continue
        velocity = np.clip(candidate, lower, upper)
        gradient = hessian @ velocity - rhs
        violation = np.where(active == -1, -gradient, np.where(active == 1, gradient, -np.inf))
        violation[fixed] = -np.inf
        release = int(np.argmax(violation))
        if violation[release] <= tolerance:
            return velocity
        active[release] = 0
    raise ArithmeticError("Bounded DLS active-set iterations exhausted")


def solve_trajectory(trajectory, q_initial, home_q, home_axes, home_anchors,
                     home_transform, joint_types=None, *, lower_limits=None,
                     upper_limits=None, settings=None):
    """Track one selected reference with CLIK and a best-effort nominal duration.

Each candidate uses midpoint integration, with decoupled position/rotation
feedback. The phase governor tries alpha=1, then searches for the largest
tested admissible alpha. Every accepted joint segment is checked at fractions
1/4, 1/2, 3/4 and 1 against T_d(tau), which is a stronger condition than merely
being close to the geometric position path. These finite checks are not a
continuous-time or physical collision/actuator guarantee.
"""
    settings = SolverSettings() if settings is None else settings
    home_q = validate_joint_angles(home_q).copy()
    q = validate_joint_angles(q_initial, len(home_q)).copy()
    axes, anchors = validate_geometry(home_axes, home_anchors, len(q))
    kinds = validate_joint_types(joint_types, len(q))
    transform_inverse(home_transform)
    geometry = (home_q, axes, anchors, home_transform, kinds)
    lower = np.full_like(q, -np.inf) if lower_limits is None else np.asarray(lower_limits, dtype=float)
    upper = np.full_like(q, np.inf) if upper_limits is None else np.asarray(upper_limits, dtype=float)
    if (lower.shape != q.shape or upper.shape != q.shape or np.any(np.isnan(lower))
            or np.any(np.isnan(upper)) or np.any(lower > upper)
            or np.any(np.isposinf(lower)) or np.any(np.isneginf(upper))):
        raise ValueError("Joint limits must have shape (n,) and lower <= upper")
    if np.any(q < lower) or np.any(q > upper):
        raise ValueError("Initial joint coordinates exceed limits")
    if not np.isfinite(trajectory.duration) or trajectory.duration <= 0:
        raise ValueError("Trajectory duration must be positive and finite")
    start, _ = sample_reference(trajectory, 0.0)
    goal, _ = sample_reference(trajectory, trajectory.duration)
    for sampled, expected in ((start, trajectory.start), (goal, trajectory.goal)):
        if max(pose_error(sampled, expected)[1:]) > 1e-8:
            raise ValueError("Trajectory samples must match start and goal metadata")
    initial_errors = pose_error(fk_manual(q, *geometry), start)[1:]
    if max(initial_errors) > 1e-8:
        raise ValueError("Trajectory start must match FK(q_initial)")
    caps = np.array([settings.revolute_speed if kind == "revolute" else settings.prismatic_speed for kind in kinds])
    weights = np.array([settings.orientation_scale] * 3 + [1.0] * 3)
    tolerances = np.array([settings.position_tolerance, settings.orientation_tolerance])
    started = perf_counter()
    times, taus, alphas, joints = [0.0], [0.0], [0.0], [q.copy()]
    guards, bounded, rejected, costs = [initial_errors], [False], [0], [0.0]
    t, tau, last_progress, previous_error = 0.0, 0.0, 0.0, 0.0
    status, message = "TIMEOUT", "Total motion time exhausted before completing the reference"

    def command(state, reference_time, alpha, velocity_lower, velocity_upper):
        current = fk_manual(state, *geometry)
        desired, nominal = sample_reference(trajectory, reference_time)
        rotation = current[:3, :3]
        relative_rotation = rotation.T @ desired[:3, :3]
        feedforward = np.r_[relative_rotation @ nominal[:3], relative_rotation @ nominal[3:]]
        feedback = np.r_[rotation_log(relative_rotation), rotation.T @ (desired[:3, 3] - current[:3, 3])]
        body = adjoint(transform_inverse(current)) @ jacobian_space(state, home_q, axes, anchors, kinds)
        return bounded_dls(weights[:, None] * body,
                           weights * (alpha * feedforward + settings.gain * feedback),
                           velocity_lower, velocity_upper, settings.damping)

    while t < settings.max_time - 1e-12:
        step_started = perf_counter()
        h = min(settings.dt, settings.max_time - t)
        velocity_lower = np.maximum(-caps, (lower - q) / h)
        velocity_upper = np.minimum(caps, (upper - q) / h)

        def candidate(alpha):
            first_velocity = command(q, tau, alpha, velocity_lower, velocity_upper)
            midpoint = q + 0.5 * h * first_velocity
            velocity = command(midpoint, min(trajectory.duration, tau + 0.5 * alpha * h),
                               alpha, velocity_lower, velocity_upper)
            trial_q = q + h * velocity
            trial_tau = min(trajectory.duration, tau + alpha * h)
            if np.any(trial_q < lower - 1e-12) or np.any(trial_q > upper + 1e-12):
                return None
            worst = np.zeros(2)
            for fraction in (0.25, 0.5, 0.75, 1.0):
                pose = fk_manual(q + fraction * (trial_q - q), *geometry)
                target, _ = sample_reference(trajectory, tau + fraction * (trial_tau - tau))
                errors = np.array(pose_error(pose, target)[1:])
                worst = np.maximum(worst, errors)
                if np.any(errors > tolerances * (1 + 1e-9)):
                    return None
            hit_bound = bool(np.any(np.isclose(velocity, velocity_lower, rtol=0, atol=1e-9))
                             or np.any(np.isclose(velocity, velocity_upper, rtol=0, atol=1e-9)))
            return trial_q, trial_tau, worst, hit_bound

        alpha_max = min(1.0, max(0.0, (trajectory.duration - tau) / h))
        rejected_count = 0
        try:
            accepted = candidate(alpha_max)
            alpha = alpha_max
            if accepted is None:
                rejected_count += 1
                accepted = candidate(0.0)
                if accepted is None:
                    # Keeping the last checked state is always admissible;
                    # never apply an unchecked correction to escape a limit.
                    errors = pose_error(fk_manual(q, *geometry), sample_reference(trajectory, tau)[0])[1:]
                    accepted = (q.copy(), tau, np.array(errors), True)
                low, high, alpha = 0.0, alpha_max, 0.0
                for _ in range(settings.phase_search_iterations):
                    trial_alpha = (low + high) / 2
                    trial = candidate(trial_alpha)
                    if trial is not None:
                        low, alpha, accepted = trial_alpha, trial_alpha, trial
                    else:
                        high = trial_alpha
                        rejected_count += 1
        except (ArithmeticError, np.linalg.LinAlgError) as exc:
            status, message = "NUMERICAL_FAILURE", str(exc)
            break
        next_q, next_tau, worst, hit_bound = accepted
        errors = np.array(pose_error(fk_manual(next_q, *geometry), sample_reference(trajectory, next_tau)[0])[1:])
        error_score = float(np.linalg.norm(errors / tolerances))
        t = min(settings.max_time, t + h)
        if next_tau - tau > 1e-8 or previous_error - error_score > 1e-5:
            last_progress = t
        q, tau, previous_error = next_q, next_tau, error_score
        times.append(t)
        taus.append(tau)
        alphas.append(alpha)
        joints.append(q.copy())
        guards.append(worst)
        bounded.append(hit_bound)
        rejected.append(rejected_count)
        costs.append(perf_counter() - step_started)
        if tau >= trajectory.duration - 1e-12 and np.all(errors <= tolerances):
            status, message = "SUCCESS", "Reference completed within the pose tolerances"
            break
        if t - last_progress >= settings.stall_time - 1e-12:
            status, message = "STALLED", "No reference progress or meaningful error reduction during the stall window"
            break
    return TrajectoryResult(np.array(times), np.array(taus), np.array(alphas), np.array(joints),
                            np.array(guards), np.array(bounded), np.array(rejected), np.array(costs),
                            status, message, perf_counter() - started)
