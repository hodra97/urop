"""Simulate the selected model; animate its optional configured gripper."""

import time

import mujoco
import mujoco.viewer

from robot_config import robot


def main() -> None:
    robot.reset()
    model, data = robot.model, robot.data
    gripper_id = robot.gripper_actuator_id
    if gripper_id is not None:
        if not model.actuator_ctrllimited[gripper_id]:
            raise ValueError("The gripper demo requires a finite actuator control range")
        closed_ctrl, open_ctrl = model.actuator_ctrlrange[gripper_id]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
    print(f"Arm joints={robot.joint_count}, nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print("Actuators:", names)
    print("Close the viewer window to stop the program.")
    if gripper_id is not None:
        print("The configured gripper opens and closes every two simulation seconds.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            if gripper_id is not None:
                data.ctrl[gripper_id] = open_ctrl if int(data.time // 2) % 2 == 0 else closed_ctrl
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
