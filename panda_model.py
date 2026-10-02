"""Load the Franka Panda MuJoCo model and capture its home geometry."""

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray


# The model repository sits beside franka_mujoco in the urop workspace.
XML_PATH: Path = (
    Path(__file__).resolve().parent.parent
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

ARM_JOINT_NAMES: tuple[str, ...] = (
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"
)
END_EFFECTOR_BODY_NAME: str = "hand"

if not XML_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {XML_PATH}")

# model holds the fixed robot description; data holds its current state.
model: mujoco.MjModel = mujoco.MjModel.from_xml_path(str(XML_PATH))
data: mujoco.MjData = mujoco.MjData(model)


def get_object_id(object_type: mujoco.mjtObj, name: str) -> int:
    """Return a MuJoCo object ID, raising a clear error if it is missing."""
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Could not find MuJoCo object: {name}")
    return object_id


home_id = get_object_id(mujoco.mjtObj.mjOBJ_KEY, "home")
hand_body_id = get_object_id(mujoco.mjtObj.mjOBJ_BODY, END_EFFECTOR_BODY_NAME)
joint_ids = np.array(
    [get_object_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES],
    dtype=int,
)

for joint_name, joint_id in zip(ARM_JOINT_NAMES, joint_ids):
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
        raise RuntimeError(f"{joint_name} is not a hinge joint.")

# Position and velocity addresses happen to match for these hinge joints.
qpos_indices = np.array([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=int)
dof_indices = np.array([model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=int)

# The named home keyframe can have nonzero joint angles.
mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)


def get_hand_transform() -> NDArray[np.float64]:
    """Read the current hand pose as a world-frame 4x4 transform."""
    transform = np.eye(4)
    transform[:3, :3] = data.xmat[hand_body_id].reshape(3, 3)
    transform[:3, 3] = data.xpos[hand_body_id]
    return transform


# MuJoCo updates data arrays in place, so copy the home geometry once.
HOME_Q: NDArray[np.float64] = data.qpos[qpos_indices].copy()
HOME_AXES: NDArray[np.float64] = data.xaxis[joint_ids].copy()
HOME_ANCHORS: NDArray[np.float64] = data.xanchor[joint_ids].copy()
HOME_HAND_TRANSFORM: NDArray[np.float64] = get_hand_transform()


def joint_limits() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the lower and upper limits in arm-joint order."""
    return model.jnt_range[joint_ids, 0].copy(), model.jnt_range[joint_ids, 1].copy()


def hand_transform_at(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ask MuJoCo for the hand pose at q without advancing simulation time."""
    data.qpos[qpos_indices] = q
    mujoco.mj_forward(model, data)
    return get_hand_transform()
