"""Animate the joint configuration selected in joint_inputs.py.

This file imports the FK calculation and MuJoCo model from franka_fk_task.py.
It reads the goal angles from joint_inputs.py. The viewer assigns joint
positions directly: it shows how the geometry moves, without simulating the
actuators following commands or solving IK along a Cartesian path.
"""

import time

import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import ArrayLike

import franka_fk_task as fk
import joint_inputs


def visualize_fk_motion(
    q_start: ArrayLike, q_goal: ArrayLike, travel_time: float = 3.0
) -> None:
    """Animate q_start -> q_goal -> q_start, repeating until the viewer closes.

    travel_time is seconds for one direction; a full round trip takes twice
    that long. The interpolation is in joint space, so the hand can trace a
    curved path even though each joint moves smoothly between two angles.
    """
    start = fk.validate_joint_angles(q_start)
    goal = fk.validate_joint_angles(q_goal)
    if not np.isfinite(travel_time) or travel_time <= 0.0:
        raise ValueError("travel_time must be a positive finite number")

    # fk.main() ends at its last test pose. Reset before opening the viewer
    # so the first displayed frame begins at the requested start pose.
    fk.data.qpos[fk.qpos_indices] = start
    fk.data.qvel[:] = 0.0
    mujoco.mj_forward(fk.model, fk.data)

    print("\nOpening MuJoCo viewer.")
    print("The arm will move between home and your chosen joint inputs.")
    print("Close the viewer window to finish.")

    cycle_time = 2.0 * travel_time  # out and back
    display_period = 1.0 / 60.0  # aim for 60 displayed frames per second

    with mujoco.viewer.launch_passive(fk.model, fk.data) as viewer:
        animation_start = time.perf_counter()

        while viewer.is_running():
            frame_start = time.perf_counter()
            cycle_position = (frame_start - animation_start) % cycle_time

            # fraction is 0 at home and 1 at the chosen pose. Reverse it for
            # the return trip so the cycle repeats without a position jump.
            if cycle_position <= travel_time:
                fraction = cycle_position / travel_time
            else:
                fraction = (cycle_time - cycle_position) / travel_time

            # Cubic smoothstep starts and ends each trip with zero joint speed.
            # displayed_q is an elementwise blend of the two angle vectors.
            blend = fraction**2 * (3.0 - 2.0 * fraction)
            displayed_q = (1.0 - blend) * start + blend * goal

            with viewer.lock():
                # The viewer may run on another thread. Lock shared data while
                # changing qpos, then recompute all visible body positions.
                fk.data.qpos[fk.qpos_indices] = displayed_q
                fk.data.qvel[:] = 0.0
                mujoco.mj_forward(fk.model, fk.data)

            viewer.sync()  # show the new robot pose in the window

            # Sleep only for display pacing; no physics time is advanced here.
            remaining = display_period - (time.perf_counter() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)


def main() -> None:
    """Validate FK, then animate the selected input configuration."""
    fk.main()  # Raises SystemExit if manual FK disagrees with MuJoCo.

    # Read the same editable angles used by the non-visualized FK checks.
    joint_min = fk.model.jnt_range[fk.joint_ids, 0]
    joint_max = fk.model.jnt_range[fk.joint_ids, 1]
    goal_q = joint_inputs.selected_joint_angles(
        fk.HOME_Q, joint_min, joint_max
    )
    visualize_fk_motion(fk.HOME_Q, goal_q)


if __name__ == "__main__":
    main()
