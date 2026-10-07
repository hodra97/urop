"""Integration checks for chain discovery, model switching, FK and Jacobians."""

import unittest
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from itertools import count
from unittest.mock import patch

import mujoco
import numpy as np
from numpy.testing import assert_allclose

import franka_fk
import franka_fk_GUI
import franka_demo
import franka_jacobian
import joint_inputs
import panda_joint_inputs
import panda_model
import ur5e_joint_inputs
import ur5e_model
from robot_model import RobotModel


def make_arm(count):
    # An unrelated free body deliberately makes qpos and velocity addresses
    # differ. A downstream finger must not enter the selected arm chain.
    xml = ['<mujoco><compiler angle="radian"/><worldbody><body name="object" pos="0 0 2">',
           '<freejoint/><geom type="sphere" size="0.02"/></body>']
    for index in range(count):
        kind = "hinge" if index % 2 else "slide"
        axis = "0 0 1" if index % 2 else "1 0 0"
        xml.append(f'<body pos="0.1 0.02 0.2" euler="0.1 0.2 0.3">'
                   f'<joint name="arm_{index}" type="{kind}" axis="{axis}" '
                   'limited="true" range="-1 1"/>'
                   '<geom type="sphere" size="0.03"/>')
    xml.append('<body name="tool" pos="0.05 0.02 0.1"><body name="finger">'
               '<joint name="finger_joint" type="slide" axis="0 1 0"/>'
               '<geom type="sphere" size="0.01"/></body></body>')
    xml.extend(['</body>'] * count)
    xml.append('</worldbody><keyframe><key name="home"/></keyframe></mujoco>')
    model = mujoco.MjModel.from_xml_string("".join(xml))
    for index in range(count):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"arm_{index}")
        model.key_qpos[0, model.jnt_qposadr[joint_id]] = 0.05 * (index + 1)
    return RobotModel(model, "tool", home_keyframe="home")


class RobotModelTests(unittest.TestCase):
    def check_kinematics(self, robot):
        with patch.object(franka_fk, "robot", robot), patch.object(franka_jacobian, "robot", robot):
            for _, q in joint_inputs.test_configurations(robot):
                assert_allclose(franka_fk.fk_manual(q), franka_fk.fk_mujoco(q),
                                atol=1e-12, rtol=0)
                manual = franka_jacobian.jacobians_manual(q)
                reference = franka_jacobian.jacobians_mujoco(q)
                self.assertEqual(len(manual), 2)
                for actual, expected in zip(manual, reference):
                    self.assertEqual(actual.shape, (6, robot.joint_count))
                    assert_allclose(actual, expected, atol=1e-12, rtol=0)

    def test_mixed_chains_with_scene_objects_and_gripper(self):
        for count in (1, 3, 6, 7, 9):
            with self.subTest(count=count):
                robot = make_arm(count)
                self.assertEqual(robot.joint_count, count)
                self.assertEqual(robot.joint_names, tuple(f"arm_{i}" for i in range(count)))
                self.assertTrue(np.all(robot.qpos_indices != robot.dof_indices))
                self.check_kinematics(robot)

    def test_real_robot_profiles(self):
        for profile, expected_count in ((panda_model, 7), (ur5e_model, 6)):
            with self.subTest(profile=profile.__name__):
                robot = profile.load_robot()
                self.assertEqual(robot.joint_count, expected_count)
                self.check_kinematics(robot)

    def test_named_offsets_vector_validation_and_limits(self):
        robot = make_arm(3)
        offsets = np.zeros(robot.joint_count)
        offsets[-1] = 0.1
        assert_allclose(joint_inputs.selected_joint_angles(robot, offsets),
                        joint_inputs.selected_joint_angles(robot, {robot.joint_names[-1]: 0.1}))
        for bad in (np.zeros(robot.joint_count + 1), {"missing": 0.1},
                    {robot.joint_names[0]: np.nan}, {robot.joint_names[0]: 5.0}):
            with self.assertRaises(ValueError):
                joint_inputs.selected_joint_angles(robot, bad)

    def test_robot_specific_inputs_are_applied_and_isolated(self):
        panda_before = joint_inputs.selected_joint_angles(panda_model.load_robot())
        ur5e_before = joint_inputs.selected_joint_angles(ur5e_model.load_robot())
        for profile, inputs, joint, other_profile, other_q in (
            (panda_model, panda_joint_inputs, "joint1", ur5e_model, ur5e_before),
            (ur5e_model, ur5e_joint_inputs, "shoulder_pan_joint", panda_model, panda_before),
        ):
            with self.subTest(profile=profile.__name__):
                before = joint_inputs.selected_joint_angles(profile.load_robot())
                with patch.dict(inputs.JOINT_OFFSETS, {joint: inputs.JOINT_OFFSETS[joint] + 0.1}):
                    robot = profile.load_robot()
                    self.assertEqual(set(inputs.JOINT_OFFSETS), set(robot.joint_names))
                    expected = before.copy()
                    expected[robot.joint_names.index(joint)] += 0.1
                    assert_allclose(joint_inputs.selected_joint_angles(robot), expected)
                    assert_allclose(joint_inputs.test_configurations(robot)[1][1], expected)
                    assert_allclose(joint_inputs.selected_joint_angles(other_profile.load_robot()), other_q)
                    self.check_kinematics(robot)

    def test_viewer_loops_accept_both_model_sizes_without_opening_windows(self):
        class Viewer:
            def __init__(self):
                self.frames = 0

            def is_running(self):
                return self.frames < 3

            def sync(self):
                self.frames += 1

            def lock(self):
                return nullcontext()

        for profile in (panda_model, ur5e_model):
            for module in (franka_fk_GUI, franka_demo):
                with self.subTest(profile=profile.__name__, module=module.__name__):
                    robot = profile.load_robot()
                    viewer = Viewer()
                    ticks = count()
                    with (patch.object(module, "robot", robot),
                          patch.object(module.mujoco.viewer, "launch_passive", return_value=nullcontext(viewer)),
                          patch.object(module.time, "sleep"),
                          patch.object(module.time, "perf_counter", side_effect=lambda: next(ticks) * 0.1),
                          redirect_stdout(StringIO())):
                        if module is franka_demo:
                            module.main()
                            self.assertGreater(robot.data.time, 0)
                        else:
                            goal = joint_inputs.selected_joint_angles(robot)
                            module.visualize_fk_motion(robot.home_q, goal)
                            self.assertFalse(np.allclose(robot.data.qpos[robot.qpos_indices], robot.home_q))
                    self.assertEqual(viewer.frames, 3)
                    self.assertTrue(np.all(np.isfinite(robot.data.qpos)))

    def test_unlimited_joint_sampling_is_finite(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="tip"><joint name="hinge"/>'
            '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>')
        robot = RobotModel(model, "tip")
        lower, upper = robot.joint_limits()
        self.assertTrue(np.isneginf(lower[0]))
        self.assertTrue(np.isposinf(upper[0]))
        for _, q in joint_inputs.test_configurations(robot):
            self.assertTrue(np.all(np.isfinite(q)))
        self.check_kinematics(robot)

    def test_viewer_reaches_every_joint_target_and_returns_home(self):
        robot = ur5e_model.load_robot()
        # Exercise all six joints, independently of editable input values.
        offsets = np.array([0.9, 2.1, -1.1, 0.6, -0.5, 0.8])
        goal = joint_inputs.selected_joint_angles(robot, offsets)
        frames = []

        class Viewer:
            def is_running(self):
                return len(frames) < 5

            def lock(self):
                return nullcontext()

            def sync(self):
                frames.append(robot.data.qpos[robot.qpos_indices].copy())

        ticks = iter([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        with (patch.object(franka_fk_GUI, "robot", robot),
              patch.object(franka_fk_GUI.mujoco.viewer, "launch_passive",
                           return_value=nullcontext(Viewer())),
              patch.object(franka_fk_GUI.time, "perf_counter", side_effect=lambda: next(ticks)),
              patch.object(franka_fk_GUI.time, "sleep"),
              redirect_stdout(StringIO())):
            franka_fk_GUI.visualize_fk_motion(robot.home_q, goal)
        expected = robot.home_q + np.array([0, 0.5, 1, 0.5, 0])[:, None] * offsets
        assert_allclose(frames, expected, atol=1e-12, rtol=0)

    def test_viewer_single_joint_option_keeps_other_coordinates_at_home(self):
        robot = ur5e_model.load_robot()
        with (patch.object(franka_fk_GUI, "robot", robot),
              patch.object(franka_fk_GUI.fk, "main"),
              patch.object(franka_fk_GUI, "visualize_fk_motion") as animate,
              redirect_stdout(StringIO())):
            for index, name in enumerate(robot.joint_names):
                franka_fk_GUI.main(["--joint", name])
                expected = robot.home_q.copy()
                expected[index] = joint_inputs.selected_joint_angles(robot)[index]
                start, goal = animate.call_args.args
                assert_allclose(start, robot.home_q)
                assert_allclose(goal, expected)

    def test_reject_unsupported_chain_and_wrong_endpoint(self):
        for kind in ("ball", "free"):
            model = mujoco.MjModel.from_xml_string(
                f'<mujoco><worldbody><body name="tip"><joint type="{kind}"/>'
                '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>')
            with self.assertRaisesRegex(ValueError, "hinge/slide"):
                RobotModel(model, "tip")
        with self.assertRaisesRegex(ValueError, "not found"):
            RobotModel(model, "unknown")
        with self.assertRaisesRegex(ValueError, "no arm joints"):
            RobotModel(model, "world")


if __name__ == "__main__":
    unittest.main()
