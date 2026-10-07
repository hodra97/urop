"""IK math, real-model verification, failures, and viewer behavior without windows."""

from contextlib import nullcontext, redirect_stdout
from dataclasses import replace
from io import StringIO
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from numpy.testing import assert_allclose

import end_effector_inputs as inputs
import franka_ik_GUI as gui
import panda_model
import ur5e_model
from FK_algorithm import fk_manual, revolute_joint_transform, skew
from IK_algorithm import pose_error, reach_bound, rotation_log, se3_log, solve_ik
from viewer_overlays import desired_pose_text, rotation_to_rpy_deg


def solve_robot(robot, target, **kwargs):
    lower, upper = robot.joint_limits()
    return solve_ik(target, robot.home_q, robot.home_q, robot.home_axes,
                    robot.home_anchors, robot.home_transform, robot.joint_types,
                    lower_limits=lower, upper_limits=upper, **kwargs)


class IKMathTests(unittest.TestCase):
    def test_pose_display_preserves_inputs_and_fk_rotation(self):
        for angles in ([130, 45, 70], [130, 135, 430], [30, 90, 80], [30, -90, 80]):
            target = inputs.target_transform([0.5, 0.56, 0.45], angles)
            canonical = rotation_to_rpy_deg(target[:3, :3])
            assert_allclose(inputs.target_transform(target[:3, 3], canonical), target, atol=1e-8)
            _, _, labels, values = desired_pose_text(target, angles)
            display = dict(zip(labels.splitlines(), values.splitlines()))
            self.assertEqual(float(display["x (m)"]), 0.5)
            self.assertEqual(float(display["y (m)"]), 0.56)
            self.assertEqual(float(display["z (m)"]), 0.45)
            assert_allclose([float(display[name]) for name in ("roll (deg)", "pitch (deg)", "yaw (deg)")], angles)

    def test_pose_input_uses_world_position_and_zyx_rpy(self):
        transform = inputs.target_transform([1, 2, 3], [90, 0, 90])
        assert_allclose(transform[:3, 3], [1, 2, 3])
        assert_allclose(transform[:3, :3], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-15)
        for position, angles in (([1, 2], [0, 0, 0]), ([1, 2, np.nan], [0, 0, 0]),
                                 ([0, 0, 0], [0, np.inf, 0])):
            with self.assertRaises(ValueError):
                inputs.target_transform(position, angles)

    def test_se3_log_round_trip_at_zero_small_and_pi_angles(self):
        for axis in ([1, 0, 0], [0, 1, 0], [1, -2, 3]):
            for theta in (0, 1e-9, 0.6, np.pi - 1e-7, np.pi, np.pi + 1e-7):
                with self.subTest(axis=axis, theta=theta):
                    transform = revolute_joint_transform(axis, [0, 0, 0], theta)
                    transform[:3, 3] = [0.3, -0.4, 0.8]
                    twist = se3_log(transform)
                    angle = np.linalg.norm(twist[:3])
                    omega = skew(twist[:3])
                    if angle < 1e-7:
                        rotation = np.eye(3) + omega + 0.5 * omega @ omega
                        left_jacobian = np.eye(3) + 0.5 * omega + omega @ omega / 6
                    else:
                        rotation = revolute_joint_transform(twist[:3], [0, 0, 0], angle)[:3, :3]
                        left_jacobian = (np.eye(3) + (1 - np.cos(angle)) / angle**2 * omega
                                         + (angle - np.sin(angle)) / angle**3 * omega @ omega)
                    assert_allclose(rotation, transform[:3, :3], atol=1e-8)
                    assert_allclose(left_jacobian @ twist[3:], transform[:3, 3], atol=1e-9)
        with self.assertRaises(ValueError):
            rotation_log(np.diag([1, 1, -1]))

    def test_single_revolute_half_turn(self):
        target = revolute_joint_transform([0, 0, 1], [0, 0, 0], np.pi)
        result = solve_ik(target, [0], [0], [[0, 0, 1]], [[0, 0, 0]], np.eye(4),
                          lower_limits=[-np.pi], upper_limits=[np.pi], restarts=0)
        self.assertTrue(result.success, result)
        self.assertAlmostEqual(abs(result.q[0]), np.pi, delta=1e-3)

    def test_mixed_revolute_prismatic_chain(self):
        home = np.eye(4)
        home[:3, 3] = [0.5, 0, 0]
        axes = [[0, 0, 1], [0, 0, 1]]
        anchors = [[0, 0, 0], [0.5, 0, 0]]
        kinds = ("revolute", "prismatic")
        q_home = np.array([0.1, 0.2])
        q_target = [0.8, 0.4]
        target = fk_manual(q_target, q_home, axes, anchors, home, kinds)
        result = solve_ik(target, q_home, q_home, axes, anchors, home, kinds,
                          lower_limits=[-2, 0], upper_limits=[2, 1], restarts=0)
        self.assertTrue(result.success, result)
        assert_allclose(result.q, q_target, atol=1e-4)

    def test_impossible_position_and_uncertain_orientation_are_distinct(self):
        kwargs = dict(q_initial=[0], home_q=[0], home_axes=[[1, 0, 0]],
                      home_anchors=[[0, 0, 0]], home_transform=np.eye(4),
                      joint_types=("prismatic",), lower_limits=[-1], upper_limits=[1],
                      restarts=1, max_iterations=10)
        result = solve_ik(inputs.target_transform([2, 0, 0], [0, 0, 0]), **kwargs)
        self.assertEqual(result.status, "IMPOSSIBLE")
        self.assertEqual(result.iterations, 0)
        result = solve_ik(inputs.target_transform([0, 0, 0], [0, 0, 90]), **kwargs)
        self.assertEqual(result.status, "NO_CONVERGENCE")
        self.assertFalse(result.success)
        # Unlimited slides must not receive a finite workspace radius.
        kwargs.update(lower_limits=[-np.inf], upper_limits=[np.inf])
        result = solve_ik(inputs.target_transform([10, 0, 0], [0, 0, 0]), **kwargs)
        self.assertNotEqual(result.status, "IMPOSSIBLE")

    def test_invalid_initial_target_limits_and_options(self):
        kwargs = dict(target=np.eye(4), q_initial=[0], home_q=[0], home_axes=[[0, 0, 1]],
                      home_anchors=[[0, 0, 0]], home_transform=np.eye(4),
                      lower_limits=[-1], upper_limits=[1])
        for override in ({"target": np.eye(3)}, {"target": np.full((4, 4), np.nan)},
                         {"q_initial": [2]}, {"q_initial": [0, 1]},
                         {"lower_limits": [2]}, {"upper_limits": [np.nan]},
                         {"damping": 0}, {"orientation_scale": -1},
                         {"position_tolerance": 0}, {"max_iterations": 0}, {"restarts": -1}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                solve_ik(**{**kwargs, **override})


class IKIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robots = [panda_model.load_robot(), ur5e_model.load_robot()]

    def assert_solution(self, robot, target, result):
        self.assertTrue(result.success, result)
        lower, upper = robot.joint_limits()
        self.assertTrue(np.all((lower <= result.q) & (result.q <= upper)))
        _, position, rotation = pose_error(robot.transform_at(result.q), target)
        self.assertLessEqual(position, 1e-4)
        self.assertLessEqual(rotation, 1e-3)

    def test_shared_default_target_for_both_robots(self):
        target = inputs.target_transform([0.45, 0.10, 0.45], [180, 0, 0])
        for robot in self.robots:
            with self.subTest(robot=robot.xml_path):
                self.assert_solution(robot, target, solve_robot(robot, target))

    def test_fk_generated_targets_and_workspace_bound(self):
        rng = np.random.default_rng(12)
        for robot in self.robots:
            lower, upper = robot.joint_limits()
            center, radius = reach_bound(robot.home_q, robot.home_anchors, robot.home_transform,
                                         robot.joint_types, lower, upper)
            for index in range(4):
                q = np.clip(robot.home_q + rng.uniform(-0.6, 0.6, robot.joint_count), lower, upper)
                target = robot.transform_at(q)
                self.assertLessEqual(np.linalg.norm(target[:3, 3] - center), radius + 1e-12)
                with self.subTest(robot=robot.xml_path, sample=index):
                    self.assert_solution(robot, target, solve_robot(robot, target))
            # Exercise full-range reach bound independently of local IK convergence.
            for q in rng.uniform(lower, upper, (20, robot.joint_count)):
                target = robot.transform_at(q)
                self.assertLessEqual(np.linalg.norm(target[:3, 3] - center), radius + 1e-12)

    def test_success_at_initial_pose_and_small_budget_failure(self):
        for robot in self.robots:
            result = solve_robot(robot, robot.home_transform)
            self.assertTrue(result.success)
            self.assertEqual(result.iterations, 0)
            target = robot.home_transform.copy()
            target[0, 3] += 0.05
            result = solve_robot(robot, target, max_iterations=1, restarts=0)
            self.assertEqual(result.status, "NO_CONVERGENCE")
            self.assertTrue(np.all(np.isfinite(result.q)))

    def test_cli_status_and_invalid_input_without_viewer(self):
        with patch.object(gui, "robot", self.robots[0]), patch.object(gui, "visualize_solution") as viewer:
            with redirect_stdout(StringIO()) as output:
                code = gui.main(["--no-viewer", "--position", "100", "0", "0"])
            self.assertEqual(code, 1)
            self.assertIn("IMPOSSIBLE", output.getvalue())
            self.assertIn("out of task-space", output.getvalue())
            with redirect_stdout(StringIO()) as output:
                code = gui.main(["--no-viewer", "--position", "nan", "0", "0"])
            self.assertEqual(code, 2)
            self.assertIn("INVALID_INPUT", output.getvalue())
            viewer.assert_not_called()

    def test_viewer_reaches_solution_holds_and_displays_failure(self):
        robot = self.robots[1]
        target = inputs.target_transform([0.45, 0.10, 0.45], [180, 0, 0])
        success = solve_robot(robot, target)
        self.assertTrue(success.success)
        failure = replace(success, success=False, status="IMPOSSIBLE", message="out of task-space")
        for result in (success, failure):
            frames, overlays = [], []

            class Viewer:
                user_scn = mujoco.MjvScene(robot.model, maxgeom=200)

                def is_running(self):
                    return len(frames) < 4

                def lock(self):
                    return nullcontext()

                def sync(self):
                    frames.append(robot.data.qpos[robot.qpos_indices].copy())

                def set_texts(self, texts):
                    overlays.append(texts)

            viewer = Viewer()
            ticks = iter([0, 0, 0, 1.5, 1.5, 3, 3, 4, 4])
            with (patch.object(gui, "robot", robot),
                  patch.object(gui.mujoco.viewer, "launch_passive", return_value=nullcontext(viewer)),
                  patch.object(gui.time, "perf_counter", side_effect=lambda: next(ticks)),
                  patch.object(gui.time, "sleep")):
                gui.visualize_solution(target, robot.home_q, result)
            goal = result.q if result.success else robot.home_q
            expected = robot.home_q + np.array([0, 0.5, 1, 1])[:, None] * (goal - robot.home_q)
            assert_allclose(frames, expected, atol=1e-12)
            self.assertGreater(viewer.user_scn.ngeom, 3)
            self.assertIn(result.status, overlays[-1][0][3])
            self.assertIn("Desired pose", overlays[-1][1][2])
            self.assertIn("0.4500", overlays[-1][1][3])


if __name__ == "__main__":
    unittest.main()
