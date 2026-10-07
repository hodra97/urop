"""Animate the joint configuration from the selected robot's input file.

This file runs the FK checks, then reads the selected model from robot_config.py
and applies its profile's offsets through joint_inputs.py. The viewer assigns joint
positions directly: it shows how the geometry moves, without simulating the
actuators following commands or solving IK along a Cartesian path.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import ArrayLike

import franka_fk as fk
import FK_algorithm
import joint_inputs
from robot_config import robot
from viewer_overlays import desired_pose_text, draw_target_axes, draw_world_grid


def visualize_fk_motion(
    q_start: ArrayLike, q_goal: ArrayLike, travel_time: float = 2.0
) -> None:
    """Animate q_start -> q_goal -> q_start, repeating until the viewer closes.

    travel_time is seconds for one direction; a full round trip takes twice
    that long. The interpolation is in joint space, so the hand can trace a
    curved path even though each joint moves smoothly between two angles.
    """
    start = FK_algorithm.validate_joint_angles(q_start, robot.joint_count)
    goal = FK_algorithm.validate_joint_angles(q_goal, robot.joint_count)
    if not np.isfinite(travel_time) or travel_time <= 0.0:
        raise ValueError("travel_time must be a positive finite number")
    target_pose = FK_algorithm.fk_manual(
        goal, robot.home_q, robot.home_axes, robot.home_anchors,
        robot.home_transform, robot.joint_types,
    )

    # fk.main() ends at its last test pose. Reset before opening the viewer
    # so the first displayed frame begins at the requested start pose.
    robot.data.qpos[robot.qpos_indices] = start
    robot.data.qvel[:] = 0.0
    mujoco.mj_forward(robot.model, robot.data)

    print("\nOpening MuJoCo viewer.")
    print(f"Model: {robot.xml_path}")
    print("Joint targets (rad for revolute joints, m for prismatic joints):")
    print(f"{'Joint':24s} {'Start':>10s} {'Target':>10s} {'Offset':>10s}")
    for name, initial, target in zip(robot.joint_names, start, goal):
        print(f"{name:24s} {initial:10.4f} {target:10.4f} {target - initial:10.4f}")
    print("The arm will move between home and your chosen joint inputs.")
    print("Close the viewer window to finish.")

    cycle_time = 2.0 * travel_time  # out and back
    display_period = 1.0 / 60.0  # aim for 60 displayed frames per second

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            draw_target_axes(viewer.user_scn, target_pose)
            draw_world_grid(viewer.user_scn)
        viewer.set_texts(desired_pose_text(target_pose))
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
                robot.data.qpos[robot.qpos_indices] = displayed_q
                robot.data.qvel[:] = 0.0
                mujoco.mj_forward(robot.model, robot.data)

            viewer.sync()  # show the new robot pose in the window

            # Sleep only for display pacing; no physics time is advanced here.
            remaining = display_period - (time.perf_counter() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)


def main(argv: list[str] | None = None) -> None:
    """Validate FK, then animate the selected input configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joint", choices=robot.joint_names,
        help="Animate only this joint's configured offset; keep other joints at home.",
    )
    args = parser.parse_args(argv)
    fk.main()  # Raises SystemExit if manual FK disagrees with MuJoCo.

    # Read the same editable angles used by the non-visualized FK checks.
    goal_q = joint_inputs.selected_joint_angles(robot)
    if args.joint is not None:
        index = robot.joint_names.index(args.joint)
        isolated_goal = robot.home_q.copy()
        isolated_goal[index] = goal_q[index]
        goal_q = isolated_goal
        print(f"\nSingle-joint check: {args.joint}; other joint coordinates stay at home.")
    visualize_fk_motion(robot.home_q, goal_q)


if __name__ == "__main__":
    main()
