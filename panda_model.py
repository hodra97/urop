"""Panda profile; the shared adapter discovers its arm chain automatically."""

from pathlib import Path

import panda_joint_inputs
from robot_model import RobotModel


XML_PATH = Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
END_EFFECTOR_BODY_NAME = "hand"
HOME_KEYFRAME = "home"
GRIPPER_ACTUATOR_NAME = "actuator8"


def load_robot() -> RobotModel:
    return RobotModel.from_xml_path(
        XML_PATH, END_EFFECTOR_BODY_NAME,
        home_keyframe=HOME_KEYFRAME,
        default_offsets=panda_joint_inputs.JOINT_OFFSETS,
        gripper_actuator=GRIPPER_ACTUATOR_NAME,
    )
