"""TidyBot profile: planar base (3 joints) and Kinova Gen3 arm (7 joints)."""

from pathlib import Path

from robot_model import RobotModel


XML_PATH = Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "stanford_tidybot" / "scene.xml"
END_EFFECTOR_BODY_NAME = "bracelet_link"
HOME_KEYFRAME = "home"
GRIPPER_ACTUATOR_NAME = "fingers_actuator"
FK_INPUT_MODULE = "tidybot_joint_inputs"


def load_robot() -> RobotModel:
    return RobotModel.from_xml_path(
        XML_PATH, END_EFFECTOR_BODY_NAME,
        home_keyframe=HOME_KEYFRAME,
        gripper_actuator=GRIPPER_ACTUATOR_NAME,
    )
