"""NumPy-only, joint-limited numerical IK for the existing serial PoE model.

Uses Log(T^-1 T_target), the body Jacobian, damped least squares, projected
backtracking, and deterministic restarts. Numerical failure is not proof of
unreachability. IMPOSSIBLE is reserved for a conservative position bound.

References:
https://modernrobotics.northwestern.edu/nu-gm-book-resource/6-2-numerical-inverse-kinematics-part-2-of-2/
https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from FK_algorithm import (
    fk_manual, skew, validate_geometry, validate_joint_angles, validate_joint_types,
)
from jacobian_algorithm import adjoint, jacobian_space, transform_inverse


@dataclass(frozen=True)
class IKResult:
    q: NDArray[np.float64]
    success: bool
    status: str
    message: str
    position_error: float
    orientation_error: float
    iterations: int
    attempts: int


def rotation_log(rotation: ArrayLike) -> NDArray[np.float64]:
    """SO(3) log as an axis-angle vector, including zero and pi rotations."""
    rotation = np.asarray(rotation, dtype=float)
    check = np.eye(4)
    if rotation.shape != (3, 3):
        raise ValueError("Rotation must have shape (3, 3)")
    check[:3, :3] = rotation
    transform_inverse(check)  # Validate finite SO(3) input.
    sine_axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]) / 2
    sine = np.linalg.norm(sine_axis)
    cosine = np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)
    theta = np.arctan2(sine, cosine)
    if theta < 1e-7:
        return sine_axis * (1 + theta**2 / 6)
    if np.pi - theta < 1e-5:
        # The eigenvector with eigenvalue 1 remains stable when sin(theta) -> 0.
        _, vectors = np.linalg.eigh((rotation + rotation.T) / 2)
        axis = vectors[:, -1]
        if np.dot(axis, sine_axis) < 0:
            axis = -axis
        return theta * axis
    return theta / sine * sine_axis


def se3_log(transform: ArrayLike) -> NDArray[np.float64]:
    """Return Log(T)^vee in [angular; linear] displacement order."""
    transform = np.asarray(transform, dtype=float)
    transform_inverse(transform)
    angular = rotation_log(transform[:3, :3])
    theta = np.linalg.norm(angular)
    omega = skew(angular)
    coefficient = (
        1 / 12 + theta**2 / 720 if theta < 1e-4
        else (1 - 0.5 * theta / np.tan(0.5 * theta)) / theta**2
    )
    linear = (np.eye(3) - 0.5 * omega + coefficient * omega @ omega) @ transform[:3, 3]
    return np.r_[angular, linear]


def pose_error(current: ArrayLike, target: ArrayLike):
    """Return body displacement, Euclidean position error (m), rotation error (rad)."""
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    transform_inverse(target)
    error = se3_log(transform_inverse(current) @ target)
    return error, float(np.linalg.norm(target[:3, 3] - current[:3, 3])), float(np.linalg.norm(error[:3]))


def reach_bound(home_q, anchors, home_transform, joint_types, lower, upper):
    """A conservative sphere, not an exact workspace or orientation test.

The endpoint is a sum of rotated inter-anchor segments and slide translations.
The triangle inequality bounds its distance from the first home anchor. An
unbounded slide disables this test. Rotational limits can only shrink reach.
"""
    points = np.vstack((anchors, home_transform[:3, 3]))
    radius = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    for index, kind in enumerate(joint_types):
        if kind == "prismatic":
            radius += max(abs(lower[index] - home_q[index]), abs(upper[index] - home_q[index]))
    return anchors[0].copy(), radius


def solve_ik(
    target: ArrayLike, q_initial: ArrayLike, home_q: ArrayLike,
    home_axes: ArrayLike, home_anchors: ArrayLike, home_transform: ArrayLike,
    joint_types=None, *, lower_limits: ArrayLike | None = None,
    upper_limits: ArrayLike | None = None, position_tolerance: float = 1e-4,
    orientation_tolerance: float = 1e-3, max_iterations: int = 250,
    restarts: int = 8, damping: float = 1e-3, orientation_scale: float = 0.3,
    seed: int = 7,
) -> IKResult:
    """Solve for absolute joint coordinates; all geometry is at home in world.

orientation_scale (m/rad) balances angular and linear residuals. Tolerances are
checked separately in metres and radians. Restarts use finite sampling windows
for unlimited joints, not artificial physical limits. Collision checking and
trajectory planning are outside this pose solver's scope.
"""
    home_q = validate_joint_angles(home_q)
    count = home_q.size
    q_initial = validate_joint_angles(q_initial, count).copy()
    axes, anchors = validate_geometry(home_axes, home_anchors, count)
    kinds = validate_joint_types(joint_types, count)
    target, home_transform = np.asarray(target, dtype=float), np.asarray(home_transform, dtype=float)
    transform_inverse(target)
    transform_inverse(home_transform)
    lower = np.full(count, -np.inf) if lower_limits is None else np.asarray(lower_limits, dtype=float)
    upper = np.full(count, np.inf) if upper_limits is None else np.asarray(upper_limits, dtype=float)
    if (lower.shape != (count,) or upper.shape != (count,) or np.any(np.isnan(lower))
            or np.any(np.isnan(upper)) or np.any(lower > upper)
            or np.any(np.isposinf(lower)) or np.any(np.isneginf(upper))):
        raise ValueError("Joint limits must have shape (n,) and lower <= upper")
    if np.any(q_initial < lower) or np.any(q_initial > upper):
        raise ValueError("Initial joint coordinates exceed limits")
    for value in (position_tolerance, orientation_tolerance, damping, orientation_scale):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Tolerances, damping, and orientation_scale must be positive and finite")
    if (not isinstance(max_iterations, int) or max_iterations < 1
            or not isinstance(restarts, int) or restarts < 0):
        raise ValueError("max_iterations must be positive and restarts nonnegative integers")

    def fk(q):
        return fk_manual(q, home_q, axes, anchors, home_transform, kinds)

    weights = np.array([orientation_scale] * 3 + [1.0] * 3)
    error, pos_error, rot_error = pose_error(fk(q_initial), target)
    best = (float(np.linalg.norm(weights * error)), q_initial.copy(), pos_error, rot_error)
    center, radius = reach_bound(home_q, anchors, home_transform, kinds, lower, upper)
    if np.linalg.norm(target[:3, 3] - center) > radius + position_tolerance:
        return IKResult(q_initial, False, "IMPOSSIBLE",
                        "out of task-space: target position exceeds the conservative reach bound",
                        pos_error, rot_error, 0, 0)

    rng = np.random.default_rng(seed)
    span = np.array([np.pi if kind == "revolute" else 0.3 for kind in kinds])
    sample_lower = np.where(np.isfinite(lower), lower, q_initial - span)
    sample_upper = np.where(np.isfinite(upper), upper, q_initial + span)
    step_cap = np.array([0.3 if kind == "revolute" else 0.03 for kind in kinds])
    total_iterations = 0
    for attempt in range(restarts + 1):
        if attempt == 0:
            q = q_initial.copy()
        elif attempt <= restarts // 2:
            q = np.clip(q_initial + rng.uniform(-1, 1, count) * span / 2, lower, upper)
        else:
            q = rng.uniform(sample_lower, sample_upper)
        for iteration in range(max_iterations + 1):
            transform = fk(q)
            error, pos_error, rot_error = pose_error(transform, target)
            score = float(np.linalg.norm(weights * error))
            if score < best[0]:
                best = (score, q.copy(), pos_error, rot_error)
            if pos_error <= position_tolerance and rot_error <= orientation_tolerance:
                return IKResult(q.copy(), True, "SUCCESS", "Target pose reached",
                                pos_error, rot_error, total_iterations, attempt + 1)
            if iteration == max_iterations:
                break
            total_iterations += 1
            body_jacobian = adjoint(transform_inverse(transform)) @ jacobian_space(q, home_q, axes, anchors, kinds)
            jacobian = weights[:, None] * body_jacobian
            residual = weights * error
            gradient = jacobian.T @ residual
            # Freeze joints at a bound if the descent direction points outside.
            free = ~(((q <= lower + 1e-10) & (gradient < 0))
                     | ((q >= upper - 1e-10) & (gradient > 0)) | (lower == upper))
            if not np.any(free):
                break
            active = jacobian[:, free]
            accepted = False
            for factor in (1.0, 10.0, 100.0):
                dq = np.zeros(count)
                dq[free] = active.T @ np.linalg.solve(
                    active @ active.T + (damping * factor)**2 * np.eye(6), residual)
                dq /= max(1.0, float(np.max(np.abs(dq) / step_cap)))
                for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                    candidate = np.clip(q + fraction * dq, lower, upper)
                    if np.linalg.norm(candidate - q) < 1e-12:
                        continue
                    candidate_error, _, _ = pose_error(fk(candidate), target)
                    if np.linalg.norm(weights * candidate_error) < score - 1e-12:
                        q = candidate
                        accepted = True
                        break
                if accepted:
                    break
            if not accepted:
                break
    return IKResult(best[1], False, "NO_CONVERGENCE",
                    "No solution found within joint limits and search budget; "
                    "the pose may be unreachable, but this is not a proof",
                    best[2], best[3], total_iterations, restarts + 1)
