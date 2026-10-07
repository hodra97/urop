"""NumPy-only derivative checks across different arm sizes and joint types.

Run: .venv/bin/python -m unittest -v test_jacobian_algorithm.py
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from FK_algorithm import fk_manual, revolute_joint_transform
from jacobian_algorithm import (
    adjoint, jacobian_body, jacobian_space, space_screw_axes, transform_inverse,
)


def twist_vector(twist_matrix):
    angular = (twist_matrix[:3, :3] - twist_matrix[:3, :3].T) / 2
    return np.r_[angular[2, 1], angular[0, 2], angular[1, 0], twist_matrix[:3, 3]]


class JacobianTests(unittest.TestCase):
    # Model sizes are test cases, not a dimension assumed by any implementation.
    JOINT_COUNTS = (1, 2, 6, 7, 9)

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.home = revolute_joint_transform([1, 2, 3], [0.2, 0.3, -0.4], 0.7)
        self.home[:3, 3] += [0.3, -0.2, 0.8]

    def geometry(self, count):
        return (self.rng.uniform(-1, 1, count), self.rng.normal(size=(count, 3)),
                self.rng.normal(size=(count, 3)))

    def test_all_columns_against_fk_central_differences(self):
        step = 1e-6
        for count in self.JOINT_COUNTS:
            for mixed in (False, True):
                with self.subTest(count=count, mixed=mixed):
                    home_q, axes, anchors = self.geometry(count)
                    types = tuple("prismatic" if mixed and i % 2 == 0 else "revolute"
                                  for i in range(count))
                    def fk(q):
                        return fk_manual(q, home_q, axes, anchors, self.home, types)
                    for q in [home_q, *self.rng.uniform(-2, 2, (4, count))]:
                        transform = fk(q)
                        inverse = np.linalg.inv(transform)
                        space = jacobian_space(q, home_q, axes, anchors, types)
                        body = jacobian_body(q, home_q, axes, anchors, self.home, types)
                        self.assertEqual(space.shape, (6, count))
                        self.assertEqual(body.shape, (6, count))
                        for index in range(count):
                            offset = np.eye(count)[index] * step
                            derivative = (fk(q + offset) - fk(q - offset)) / (2 * step)
                            assert_allclose(space[:, index], twist_vector(derivative @ inverse),
                                            atol=2e-8, rtol=0)
                            assert_allclose(body[:, index], twist_vector(inverse @ derivative),
                                            atol=2e-8, rtol=0)

    def test_nonzero_home_gives_home_screws(self):
        for count in self.JOINT_COUNTS:
            home_q, axes, anchors = self.geometry(count)
            unit_axes = axes / np.linalg.norm(axes, axis=1)[:, None]
            expected = np.vstack((unit_axes.T, -np.cross(unit_axes, anchors).T))
            assert_allclose(jacobian_space(home_q, home_q, axes, anchors), expected, atol=1e-14)
            assert_allclose(fk_manual(home_q, home_q, axes, anchors, self.home), self.home, atol=1e-14)

    def test_space_columns_ignore_own_and_later_coordinates(self):
        for count in self.JOINT_COUNTS:
            home_q, axes, anchors = self.geometry(count)
            types = tuple("revolute" if i % 2 else "prismatic" for i in range(count))
            baseline = jacobian_space(home_q, home_q, axes, anchors, types)
            for index in range(count):
                q = home_q.copy()
                q[index:] += self.rng.normal(size=count - index)
                result = jacobian_space(q, home_q, axes, anchors, types)
                assert_allclose(result[:, index], baseline[:, index], atol=1e-14)

    def test_prismatic_joint_has_translation_only(self):
        home_q = np.array([0.2])
        axes, anchors = [[0, 0, 2]], [[3, 4, 5]]
        types = ("prismatic",)
        result = fk_manual([0.5], home_q, axes, anchors, np.eye(4), types)
        assert_allclose(result[:3, 3], [0, 0, 0.3])
        assert_allclose(result[:3, :3], np.eye(3))
        assert_allclose(space_screw_axes(axes, anchors, types)[:, 0], [0, 0, 0, 0, 0, 1])

    def test_inverse_and_adjoint_change_twist_coordinates(self):
        assert_allclose(transform_inverse(self.home), np.linalg.inv(self.home), atol=1e-14)
        twist = np.array([[0, -3, 2, 4], [3, 0, -1, 5], [-2, 1, 0, 6], [0, 0, 0, 0]])
        expected = twist_vector(self.home @ twist @ np.linalg.inv(self.home))
        assert_allclose(adjoint(self.home) @ np.arange(1, 7), expected, atol=1e-14)

    def test_reject_invalid_inputs(self):
        count = 3
        home_q, axes, anchors = self.geometry(count)
        for bad_q in ([], np.zeros(count - 1), np.zeros((count, 1)), np.full(count, np.nan)):
            for function in (lambda q: jacobian_space(q, home_q, axes, anchors),
                             lambda q: fk_manual(q, home_q, axes, anchors, self.home)):
                with self.assertRaises(ValueError):
                    function(bad_q)
        for bad_axes in (np.zeros((count, 3)), np.full((count, 3), np.inf),
                         np.zeros((count + 1, 3))):
            with self.assertRaises(ValueError):
                space_screw_axes(bad_axes, anchors)
        for types in (("revolute",), ("ball",) * count):
            with self.assertRaises(ValueError):
                jacobian_space(home_q, home_q, axes, anchors, types)
            with self.assertRaises(ValueError):
                fk_manual(home_q, home_q, axes, anchors, self.home, types)
        bad_transform = np.eye(4)
        bad_transform[0, 0] = -1
        for bad in (np.eye(3), np.full((4, 4), np.nan), bad_transform):
            with self.assertRaises(ValueError):
                adjoint(bad)


if __name__ == "__main__":
    unittest.main()
