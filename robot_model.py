"""MuJoCo adapter for a serial arm discovered from its end-effector body.

Only joints on the world-to-end-effector path are selected. Gripper descendants
and unrelated scene objects are excluded. Hinge and slide joints are supported.
"""

from collections.abc import Mapping
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from fk_algorithm import validate_joint_angles


class RobotModel:
    def __init__(self, model: mujoco.MjModel, end_effector_body: str, *,
                 home_keyframe: str | None = None,
                 default_offsets: Mapping[str, float] | None = None,
                 gripper_actuator: str | None = None) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.xml_path: Path | None = None
        self.end_effector_body_name = end_effector_body
        self.end_effector_body_id = self.object_id(mujoco.mjtObj.mjOBJ_BODY, end_effector_body)
        self.home_keyframe = home_keyframe

        # Follow ancestors, rather than counting all scene joints/actuators.
        body_chain = []
        body_id = self.end_effector_body_id
        while body_id != 0:
            if model.body_mocapid[body_id] >= 0:
                raise ValueError("The selected arm must have a fixed base, not a mocap ancestor")
            body_chain.append(body_id)
            body_id = int(model.body_parentid[body_id])
        joint_ids = []
        for body_id in reversed(body_chain):
            first = int(model.body_jntadr[body_id])
            joint_ids.extend(range(first, first + int(model.body_jntnum[body_id])))
        if not joint_ids:
            raise ValueError("The selected end effector has no arm joints")
        self.joint_ids = np.array(joint_ids, dtype=int)
        types = {int(mujoco.mjtJoint.mjJNT_HINGE): "revolute", int(mujoco.mjtJoint.mjJNT_SLIDE): "prismatic"}
        if any(model.jnt_type[joint_id] not in types for joint_id in joint_ids):
            raise ValueError("The arm chain supports hinge/slide joints only; free/ball joints are unsupported")
        self.joint_types = tuple(types[model.jnt_type[joint_id]] for joint_id in joint_ids)
        self.joint_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_id_{joint_id}"
            for joint_id in joint_ids
        )
        self.joint_count = len(joint_ids)
        self.qpos_indices = model.jnt_qposadr[self.joint_ids].copy()
        self.dof_indices = model.jnt_dofadr[self.joint_ids].copy()
        self.default_offsets = dict(default_offsets or {})
        unknown = self.default_offsets.keys() - set(self.joint_names)
        if unknown:
            raise ValueError(f"Offsets refer to joints outside the arm: {sorted(unknown)}")
        if not all(np.isfinite(value) for value in self.default_offsets.values()):
            raise ValueError("Default offsets must be finite")
        self.gripper_actuator_id = (
            None if gripper_actuator is None
            else self.object_id(mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_actuator)
        )
        self.reset()
        self.home_q = self.data.qpos[self.qpos_indices].copy()
        self.home_axes = self.data.xaxis[self.joint_ids].copy()
        self.home_anchors = self.data.xanchor[self.joint_ids].copy()
        self.home_transform = self.end_effector_transform()

    @classmethod
    def from_xml_path(cls, xml_path: str | Path, end_effector_body: str, **kwargs):
        path = Path(xml_path).resolve()
        instance = cls(mujoco.MjModel.from_xml_path(str(path)), end_effector_body, **kwargs)
        instance.xml_path = path
        return instance

    def object_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Model object not found: {name}")
        return object_id

    def reset(self) -> None:
        if self.home_keyframe is None:
            mujoco.mj_resetData(self.model, self.data)
        else:
            key_id = self.object_id(mujoco.mjtObj.mjOBJ_KEY, self.home_keyframe)
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    def end_effector_transform(self) -> NDArray[np.float64]:
        transform = np.eye(4)
        transform[:3, :3] = self.data.xmat[self.end_effector_body_id].reshape(3, 3)
        transform[:3, 3] = self.data.xpos[self.end_effector_body_id]
        return transform

    def transform_at(self, q: ArrayLike) -> NDArray[np.float64]:
        self.data.qpos[self.qpos_indices] = validate_joint_angles(q, self.joint_count)
        mujoco.mj_forward(self.model, self.data)
        return self.end_effector_transform()

    def joint_limits(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        limits = self.model.jnt_range[self.joint_ids].copy()
        unlimited = ~self.model.jnt_limited[self.joint_ids].astype(bool)
        limits[unlimited] = [-np.inf, np.inf]
        return limits[:, 0], limits[:, 1]