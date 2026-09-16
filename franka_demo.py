import time
from pathlib import Path

import mujoco
import mujoco.viewer


XML_PATH = (
    Path.home()
    / "urop"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

if not XML_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {XML_PATH}")

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

home_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_KEY, "home"
)
if home_id < 0:
    raise RuntimeError("Could not find the home keyframe.")

mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)

gripper_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
)
if gripper_id < 0:
    raise RuntimeError("Could not find actuator8.")

closed_ctrl = float(model.actuator_ctrlrange[gripper_id, 0])
open_ctrl = float(model.actuator_ctrlrange[gripper_id, 1])

actuator_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    for actuator_id in range(model.nu)
]

print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
print("Actuators:", actuator_names)
print("The gripper opens and closes every two simulation seconds.")
print("Close the viewer window to stop the program.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.perf_counter()

        if int(data.time // 2) % 2 == 0:
            data.ctrl[gripper_id] = open_ctrl
        else:
            data.ctrl[gripper_id] = closed_ctrl

        mujoco.mj_step(model, data)
        viewer.sync()

        remaining = model.opt.timestep - (time.perf_counter() - step_start)
        if remaining > 0:
            time.sleep(remaining)
