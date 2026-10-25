"""Select profiles and explicitly create independent robot instances.

ROBOT_MODULE is read when load_profile() is called, with Panda as the default.
Importing this module does not select a profile, load XML, or read task inputs.
"""

import importlib
import os
from types import ModuleType

from robot_model import RobotModel


DEFAULT_ROBOT_MODULE = "panda_model"


def load_profile(module_name: str | None = None) -> ModuleType:
    """Import the selected profile without constructing a model."""
    if module_name is None:
        module_name = os.environ.get("ROBOT_MODULE", DEFAULT_ROBOT_MODULE)
    return importlib.import_module(module_name)


def load_robot(profile: ModuleType | None = None) -> RobotModel:
    """Create a fresh model/data pair; FK inputs are loaded separately by callers."""
    if profile is None:
        profile = load_profile()
    robot = profile.load_robot()
    if not isinstance(robot, RobotModel):
        raise TypeError(f"{profile.__name__}.load_robot() must return RobotModel")
    return robot
