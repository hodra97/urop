"""Select a robot profile for every task, viewer, and input helper.

Set ROBOT_MODULE=ur5e_model to switch without editing the algorithms.
New profiles provide load_robot() returning RobotModel, with offsets loaded
from their own joint input module (e.g. ur5e_joint_inputs.py).
"""

import importlib
import os

from robot_model import RobotModel


ROBOT_MODULE = os.environ.get("ROBOT_MODULE", "panda_model")
robot = importlib.import_module(ROBOT_MODULE).load_robot()
if not isinstance(robot, RobotModel):
    raise TypeError(f"{ROBOT_MODULE}.load_robot() must return RobotModel")
