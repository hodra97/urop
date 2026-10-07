"""Modern Robotics space/body Jacobians for an n-joint serial PoE chain.

Twists use [omega_x, omega_y, omega_z, v_x, v_y, v_z] order.
The space frame is world; the body frame is the configured end-effector origin.
All computation here is independent of MuJoCo.
Reference: https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-1-1-space-jacobian/
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fk_algorithm import (
    fk_manual, joint_transform, skew, validate_geometry,
    validate_joint_angles, validate_joint_types,
)


def _finite_array(values: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and contain finite numbers.")
    return array


def _transform(values: ArrayLike) -> NDArray[np.float64]:
    transform = _finite_array(values, (4, 4), "Transform")
    rotation = transform[:3, :3]
    if not (
        np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0)
    ):
        raise ValueError("Transform must belong to SE(3).")
    return transform


def transform_inverse(transform: ArrayLike) -> NDArray[np.float64]:
    """Return T^-1 = [[R.T, -R.T @ p], [0, 1]] for T in SE(3)."""
    transform = _transform(transform)
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def adjoint(transform: ArrayLike) -> NDArray[np.float64]:
    """Return Ad_T = [[R, 0], [[p]x @ R, R]] for [omega; v] twists.

    If T = T_ab maps coordinates from frame b to a, V_a = Ad_T @ V_b.
    """
    transform = _transform(transform)
    rotation = transform[:3, :3]
    result = np.zeros((6, 6))
    result[:3, :3] = rotation
    result[3:, :3] = skew(transform[:3, 3]) @ rotation
    result[3:, 3:] = rotation
    return result


def space_screw_axes(home_axes: ArrayLike, home_anchors: ArrayLike,
                     joint_types=None) -> NDArray[np.float64]:
    """Build (6, n) screws: revolute [axis; -axis x r], prismatic [0; axis].

    Geometry is expressed in world coordinates at the reference configuration.
    """
    axes = np.asarray(home_axes, dtype=float)
    if axes.ndim != 2 or axes.shape[0] == 0:
        raise ValueError("Home axes must have shape (n, 3), with n > 0")
    axes, anchors = validate_geometry(axes, home_anchors, axes.shape[0])
    types = validate_joint_types(joint_types, axes.shape[0])
    omega = axes.copy()
    linear = -np.cross(omega, anchors)
    for index, joint_type in enumerate(types):
        if joint_type == "prismatic":
            omega[index] = 0
            linear[index] = axes[index]
    return np.vstack((omega.T, linear.T))


def jacobian_space(
    q: ArrayLike,
    home_q: ArrayLike,
    home_axes: ArrayLike,
    home_anchors: ArrayLike,
    joint_types=None,
) -> NDArray[np.float64]:
    """Return J_s (6, n), satisfying [V_s] = T_dot @ T^-1.

    theta = q - home_q; since home_q is constant, theta_dot = q_dot.
    Column i is Ad_(exp([S_1]theta_1)...exp([S_(i-1)]theta_(i-1))) S_i.
    J_s @ q_dot is a space twist: its lower half is NOT hand position rate.
    """
    home_q = validate_joint_angles(home_q)
    joint_count = home_q.size
    q = validate_joint_angles(q, joint_count)
    axes, anchors = validate_geometry(home_axes, home_anchors, joint_count)
    joint_types = validate_joint_types(joint_types, joint_count)
    screws = space_screw_axes(axes, anchors, joint_types)
    delta = q - home_q
    jacobian = np.empty((6, joint_count))
    prefix = np.eye(4)

    for index in range(joint_count):
        # 현재 관절의 지수변환을 곱하기 전에 열을 계산한다.
        # prefix에는 앞선 관절 1, ..., i-1의 운동만 들어 있다.
        jacobian[:, index] = adjoint(prefix) @ screws[:, index]
        if index < joint_count - 1:
            prefix = prefix @ joint_transform(
                axes[index], anchors[index], delta[index], joint_types[index]
            )

    return jacobian


def jacobian_body(
    q: ArrayLike,
    home_q: ArrayLike,
    home_axes: ArrayLike,
    home_anchors: ArrayLike,
    home_hand_transform: ArrayLike,
    joint_types=None,
) -> NDArray[np.float64]:
    """Return J_b = Ad_(T^-1) J_s, so [V_b] = T^-1 @ T_dot.

    This is the Modern Robotics space/body identity, using the existing FK.
    The lower three rows map q_dot to R.T @ p_dot in the hand frame.
    """
    home_hand_transform = _transform(home_hand_transform)
    space = jacobian_space(q, home_q, home_axes, home_anchors, joint_types)
    transform = fk_manual(q, home_q, home_axes, home_anchors, home_hand_transform, joint_types)
    return adjoint(transform_inverse(transform)) @ space
