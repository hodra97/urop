"""MuJoCo-independent PoE FK for a serial arm with n scalar joints.

Revolute coordinates are radians; prismatic coordinates are metres.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_joint_angles(q: ArrayLike, joint_count: int | None = None) -> NDArray[np.float64]:
    """Validate a nonempty joint-coordinate vector, optionally against its model."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or q.size == 0:
        raise ValueError(f"Expected a nonempty 1-D joint vector, got {q.shape}")
    if joint_count is not None and q.shape != (joint_count,):
        raise ValueError(f"Expected q shape ({joint_count},), got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("Joint coordinates must be finite numbers.")
    return q


def validate_joint_types(joint_types, joint_count: int) -> tuple[str, ...]:
    """Default to revolute joints for callers supplying rotation-only geometry."""
    types = ("revolute",) * joint_count if joint_types is None else tuple(joint_types)
    if len(types) != joint_count or any(t not in ("revolute", "prismatic") for t in types):
        raise ValueError("Provide one 'revolute' or 'prismatic' type per joint")
    return types


def validate_geometry(home_axes: ArrayLike, home_anchors: ArrayLike, joint_count: int):
    """Return unit axes and anchors for the selected serial chain."""
    axes = np.asarray(home_axes, dtype=float)
    anchors = np.asarray(home_anchors, dtype=float)
    if axes.shape != (joint_count, 3) or anchors.shape != (joint_count, 3):
        raise ValueError(f"Axes and anchors must each have shape ({joint_count}, 3)")
    if not np.all(np.isfinite(axes)) or not np.all(np.isfinite(anchors)):
        raise ValueError("Axes and anchors must contain finite numbers")
    lengths = np.linalg.norm(axes, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("Joint axis has near-zero length")
    return axes / lengths[:, None], anchors


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


def joint_transform(axis: ArrayLike, anchor: ArrayLike, displacement: float,
                    joint_type: str = "revolute") -> NDArray[np.float64]:
    """Evaluate exp([S] displacement) for a revolute or prismatic joint."""
    if joint_type == "revolute":
        return revolute_joint_transform(axis, anchor, displacement)
    if joint_type != "prismatic":
        raise ValueError(f"Unsupported joint type: {joint_type}")
    axes, _ = validate_geometry(np.asarray(axis)[None, :], np.asarray(anchor)[None, :], 1)
    transform = np.eye(4)
    transform[:3, 3] = axes[0] * displacement
    return transform


def fk_manual(
    q: ArrayLike,
    home_q: ArrayLike,
    home_axes: ArrayLike,
    home_anchors: ArrayLike,
    home_hand_transform: ArrayLike,
    joint_types=None,
) -> NDArray[np.float64]:
    """Compute the hand pose from home geometry, without calling MuJoCo.

    All axes and anchors are expressed in the world frame at the home pose.
    T(q) = T_joint1 ... T_jointN @ home_hand_transform.
    """
    home_q = validate_joint_angles(home_q)
    joint_count = home_q.size
    q = validate_joint_angles(q, joint_count)
    home_axes, home_anchors = validate_geometry(home_axes, home_anchors, joint_count)
    joint_types = validate_joint_types(joint_types, joint_count)
    home_hand_transform = np.asarray(home_hand_transform, dtype=float)
    if home_hand_transform.shape != (4, 4) or not np.all(np.isfinite(home_hand_transform)):
        raise ValueError("Home end-effector transform must be finite with shape (4, 4)")

    cumulative_motion = np.eye(4)
    for index in range(joint_count):
        delta_angle = q[index] - home_q[index]
        joint_motion = joint_transform(
            home_axes[index], home_anchors[index], delta_angle, joint_types[index]
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
