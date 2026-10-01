"""Compute the Panda hand pose from seven joint angles and check it in MuJoCo.

Input: seven arm-joint angles in radians. Output: a 4x4 transform describing
the hand frame relative to the world frame. Its upper-left 3x3 block is the
hand orientation; its upper-right column is the hand position in metres.

Change the example angles in joint_inputs.py. This file contains the geometry
calculation and compares its result with MuJoCo's result.
"""

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

import joint_inputs


# The model repository sits beside franka_mujoco in the urop workspace.
XML_PATH: Path = (
    Path(__file__).resolve().parent.parent
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

ARM_JOINT_NAMES: tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
)

END_EFFECTOR_BODY_NAME: str = "hand"
POSITION_TOLERANCE: float = 1e-6  # metres
ORIENTATION_TOLERANCE: float = 1e-6  # radians


if not XML_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {XML_PATH}")

# model holds the robot's fixed description; data holds its current state.
model: mujoco.MjModel = mujoco.MjModel.from_xml_path(str(XML_PATH))
data: mujoco.MjData = mujoco.MjData(model)


def get_object_id(object_type: mujoco.mjtObj, name: str) -> int:
    """Return a MuJoCo object ID, raising a clear error if it is missing."""
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Could not find MuJoCo object: {name}")
    return object_id


home_id = get_object_id(mujoco.mjtObj.mjOBJ_KEY, "home")
hand_body_id = get_object_id(
    mujoco.mjtObj.mjOBJ_BODY,
    END_EFFECTOR_BODY_NAME,
)

joint_ids = np.array(
    [
        get_object_id(mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ARM_JOINT_NAMES
    ],
    dtype=int,
)

for joint_name, joint_id in zip(ARM_JOINT_NAMES, joint_ids):
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
        raise RuntimeError(f"{joint_name} is not a hinge joint.")

# qpos addresses locate joint positions. DoF addresses locate velocities and
# Jacobian columns. They are equal for these hinge joints, but keeping the two
# concepts separate prevents indexing errors in more complicated models.
qpos_indices = np.array(
    [model.jnt_qposadr[joint_id] for joint_id in joint_ids],
    dtype=int,
)

dof_indices = np.array(
    [model.jnt_dofadr[joint_id] for joint_id in joint_ids],
    dtype=int,
)

# "home" is a named keyframe from the Panda XML, not necessarily seven zeros.
# mj_forward computes body poses at that keyframe without advancing time.
mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)


def get_hand_transform_from_mujoco() -> NDArray[np.float64]:
    """Assemble T = [[R, p], [0, 1]] from MuJoCo's current hand pose."""
    transform = np.eye(4)
    # xmat is the hand's 3x3 orientation R, stored as nine numbers.
    transform[:3, :3] = data.xmat[hand_body_id].reshape(3, 3)
    # xpos is the hand-frame origin p, expressed in world coordinates.
    transform[:3, 3] = data.xpos[hand_body_id]
    return transform


# Save reference geometry once. HOME_AXES are rotation directions and
# HOME_ANCHORS are points on the corresponding rotation axes, all expressed
# in world coordinates at home. MuJoCo updates data arrays in place, so copy.
HOME_Q: NDArray[np.float64] = data.qpos[qpos_indices].copy()
HOME_AXES: NDArray[np.float64] = data.xaxis[joint_ids].copy()
HOME_ANCHORS: NDArray[np.float64] = data.xanchor[joint_ids].copy()
HOME_HAND_TRANSFORM: NDArray[np.float64] = get_hand_transform_from_mujoco()


def validate_joint_angles(q: ArrayLike) -> NDArray[np.float64]:
    """Require seven finite arm angles; the two finger joints are excluded."""
    q = np.asarray(q, dtype=float)
    if q.shape != (7,):
        raise ValueError(f"Expected q shape (7,), got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("Joint angles must be finite numbers.")
    return q


def fk_mujoco(q: ArrayLike) -> NDArray[np.float64]:
    """Ask MuJoCo for the hand pose at q, to check the manual result.

    This changes data.qpos and recomputes geometry; it does not simulate motion.
    """
    q = validate_joint_angles(q)

    data.qpos[qpos_indices] = q
    mujoco.mj_forward(model, data)
    return get_hand_transform_from_mujoco()


def skew(vector: ArrayLike) -> NDArray[np.float64]:
    """Return [v]x, a matrix satisfying [v]x @ w == cross(v, w)."""
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"Expected vector shape (3,), got {vector.shape}")

    x, y, z = vector
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def revolute_joint_transform(
    axis: ArrayLike, anchor: ArrayLike, angle: float
) -> NDArray[np.float64]:
    """Return a rotation about a world-space axis passing through an anchor.

    ``axis`` gives the rotation direction; ``anchor`` is a point on its line.
    Rodrigues' formula gives R = I + sin(angle)[axis]x +
    (1 - cos(angle))[axis]x². The 4x4 result also needs a translation so
    that the anchor stays fixed while the rest of the robot rotates.
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

    # R @ anchor + translation == anchor, so rotation is about this joint's
    # location instead of an axis passing through the world origin.
    translation = (identity_3 - rotation) @ anchor

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def fk_manual(q: ArrayLike) -> NDArray[np.float64]:
    """Return the hand pose using product-of-exponentials forward kinematics.

    This implementation uses only the geometry saved at the home configuration.
    It does not call MuJoCo's kinematics functions or read its current body pose.
    T(q) = T_joint1 ... T_joint7 @ HOME_HAND_TRANSFORM; at home, every
    T_joint is the identity transform, so the result is the home hand pose.
    """
    q = validate_joint_angles(q)

    cumulative_motion = np.eye(4)

    for index in range(7):
        # The saved axes and anchors describe the home pose, so each rotation
        # is measured relative to HOME_Q rather than relative to zero.
        delta_angle = q[index] - HOME_Q[index]

        joint_motion = revolute_joint_transform(
            HOME_AXES[index],
            HOME_ANCHORS[index],
            delta_angle,
        )

        # Multiplication in chain order carries later joints along when an
        # earlier joint rotates. All axes/anchors above were saved at home.
        cumulative_motion = cumulative_motion @ joint_motion

    # Carry the hand's home frame through all seven joint rotations.
    return cumulative_motion @ HOME_HAND_TRANSFORM


def rotation_error_angle(
    rotation_a: NDArray[np.float64], rotation_b: NDArray[np.float64]
) -> float:
    """Return orientation disagreement in radians (zero means equal)."""
    rotation_difference = rotation_a @ rotation_b.T
    # A rotation matrix's trace determines its rotation angle. Clamp the
    # cosine because floating-point roundoff can place it just outside [-1, 1].
    cosine_angle = (np.trace(rotation_difference) - 1.0) / 2.0
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.arccos(cosine_angle))


def compare_fk(q: ArrayLike, label: str) -> tuple[float, float]:
    """Compare both hand poses; report distance in metres and angle in radians."""
    expected = fk_mujoco(q)
    calculated = fk_manual(q)

    position_error = float(
        np.linalg.norm(expected[:3, 3] - calculated[:3, 3])
    )
    orientation_error = rotation_error_angle(
        expected[:3, :3],
        calculated[:3, :3],
    )

    print(f"\n{label}")
    print("q:", np.round(q, 4))
    print("MuJoCo position:   ", np.round(expected[:3, 3], 6))
    print("Manual FK position:", np.round(calculated[:3, 3], 6))
    print(f"Position error:    {position_error:.3e} m")
    print(f"Orientation error: {orientation_error:.3e} rad")

    return position_error, orientation_error


def main() -> None:
    np.set_printoptions(precision=5, suppress=True)

    print(f"Model: {XML_PATH}")
    print("qpos indices:", qpos_indices)
    print("DoF indices: ", dof_indices)
    print("Home joint angles:", HOME_Q)
    print("\nHome hand transformation:")
    print(HOME_HAND_TRANSFORM)

    # The input module builds the home, user-chosen, and random configurations.
    joint_min = model.jnt_range[joint_ids, 0]
    joint_max = model.jnt_range[joint_ids, 1]
    test_results = [
        compare_fk(q, label)
        for label, q in joint_inputs.test_configurations(
            HOME_Q, joint_min, joint_max
        )
    ]

    worst_position_error = max(result[0] for result in test_results)
    worst_orientation_error = max(result[1] for result in test_results)

    print("\nFinal result")
    print(f"Worst position error:    {worst_position_error:.3e} m")
    print(f"Worst orientation error: {worst_orientation_error:.3e} rad")

    passed = (
        worst_position_error < POSITION_TOLERANCE
        and worst_orientation_error < ORIENTATION_TOLERANCE
    )

    if passed:
        print("PASS: Manual FK agrees with MuJoCo.")
    else:
        print("FAIL: Manual FK does not agree with MuJoCo.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
