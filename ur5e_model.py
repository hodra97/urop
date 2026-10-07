"""An alternative one-arm profile demonstrating automatic chain dimensions."""

from pathlib import Path

import ur5e_joint_inputs
from robot_model import RobotModel


XML_PATH = Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "universal_robots_ur5e" / "scene.xml"
END_EFFECTOR_BODY_NAME = "wrist_3_link"


def load_robot() -> RobotModel:
    return RobotModel.from_xml_path(
        XML_PATH, END_EFFECTOR_BODY_NAME, home_keyframe="home",
        default_offsets=ur5e_joint_inputs.JOINT_OFFSETS,
    )
