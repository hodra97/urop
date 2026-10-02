"""MuJoCo-independent product-of-exponentials FK for a seven-joint arm."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_joint_angles(q: ArrayLike) -> NDArray[np.float64]:
    """Require seven finite arm angles; the two finger joints are excluded."""
    q = np.asarray(q, dtype=float)
    if q.shape != (7,):
        raise ValueError(f"Expected q shape (7,), got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("Joint angles must be finite numbers.")
    return q


def skew(vector: ArrayLike) -> NDArray[np.float64]:
    """Return [v]x, a matrix satisfying [v]x @ w == cross(v, w)."""
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"Expected vector shape (3,), got {vector.shape}")

    x, y, z = vector
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
    )


def revolute_joint_transform(
    axis: ArrayLike, anchor: ArrayLike, angle: float
) -> NDArray[np.float64]:
    """Return a rotation about a world-space axis passing through an anchor.

    Rodrigues' formula gives R = I + sin(angle)[axis]x +
    (1 - cos(angle))[axis]x². Translation keeps the anchor fixed.
    """
    axis = np.asarray(axis, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    if axis.shape != (3,) or anchor.shape != (3,):
        raise ValueError("axis and anchor must both have shape (3,)")

    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        raise ValueError("Rotation axis has near-zero length")

    unit_axis = axis / axis_norm
    axis_skew = skew(unit_axis)
    identity_3 = np.eye(3)
    rotation = (
        identity_3
        + np.sin(angle) * axis_skew
        + (1.0 - np.cos(angle)) * (axis_skew @ axis_skew)
    )
    translation = (identity_3 - rotation) @ anchor

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def fk_manual(
    q: ArrayLike,
    home_q: ArrayLike,
    home_axes: ArrayLike,
    home_anchors: ArrayLike,
    home_hand_transform: ArrayLike,
) -> NDArray[np.float64]:
    """Compute the hand pose from home geometry, without calling MuJoCo.

    All axes and anchors are expressed in the world frame at the home pose.
    T(q) = T_joint1 ... T_joint7 @ home_hand_transform.
    """
    q = validate_joint_angles(q)
    home_q = validate_joint_angles(home_q)
    home_axes = np.asarray(home_axes, dtype=float)
    home_anchors = np.asarray(home_anchors, dtype=float)
    home_hand_transform = np.asarray(home_hand_transform, dtype=float)
    if home_axes.shape != (7, 3) or home_anchors.shape != (7, 3):
        raise ValueError("Home axes and anchors must each have shape (7, 3)")
    if home_hand_transform.shape != (4, 4):
        raise ValueError("Home hand transform must have shape (4, 4)")

    cumulative_motion = np.eye(4)
    for index in range(7):
        delta_angle = q[index] - home_q[index]
        joint_motion = revolute_joint_transform(
            home_axes[index], home_anchors[index], delta_angle
        )
        cumulative_motion = cumulative_motion @ joint_motion

    return cumulative_motion @ home_hand_transform


def rotation_error_angle(
    rotation_a: NDArray[np.float64], rotation_b: NDArray[np.float64]
) -> float:
    """Return orientation disagreement in radians (zero means equal)."""
    rotation_difference = rotation_a @ rotation_b.T
    cosine_angle = (np.trace(rotation_difference) - 1.0) / 2.0
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.arccos(cosine_angle))
