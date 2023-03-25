import unittest
import numpy as np
from quaternion import Quaternion, Line, Plane
from typing import TypeVar, Union

class TestQuaternion(unittest.TestCase):
    def test_add(self):
        q1 = Quaternion(0, 1, 2, 3)
        q2 = Quaternion(0, 4, 5, 6)
        result = q1.add(q2)
        self.assertEqual(result, Quaternion(0, 5, 7, 9))

    def test_dot_product(self):
        q1 = Quaternion(0, 1, 2, 3)
        q2 = Quaternion(0, 4, 5, 6)
        result = q1.dot_product(q2)
        self.assertEqual(result, 32)

    def test_cross_product(self):
        q1 = Quaternion(0, 1, 2, 3)
        q2 = Quaternion(0, 4, 5, 6)
        result = q1.cross_product(q2)
        self.assertEqual(result, Quaternion(0, -3, 6, -3))

    def test_slerp(self):
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion(0, 1, 0, 0)

        slerp = Quaternion.slerp(q1, q2, 0.0)
        expected_slerp = q1
        self.assertAlmostEqual(slerp.w, expected_slerp.w, "w")
        self.assertAlmostEqual(slerp.x, expected_slerp.x, "x")
        self.assertAlmostEqual(slerp.y, expected_slerp.y, "y")
        self.assertAlmostEqual(slerp.z, expected_slerp.z, "z")

        slerp = Quaternion.slerp(q1, q2, 0.2)
        expected_slerp = Quaternion(0.951056516295154, 0.309016994374947, 0.0, 0.0)
        self.assertAlmostEqual(slerp.w, expected_slerp.w)
        self.assertAlmostEqual(slerp.x, expected_slerp.x)
        self.assertAlmostEqual(slerp.y, expected_slerp.y)
        self.assertAlmostEqual(slerp.z, expected_slerp.z)

        slerp = Quaternion.slerp(q1, q2, 0.4)
        expected_slerp = Quaternion(0.809016994374947, 0.587785252292473, 0.0, 0.0)
        self.assertAlmostEqual(slerp.w, expected_slerp.w)
        self.assertAlmostEqual(slerp.x, expected_slerp.x)
        self.assertAlmostEqual(slerp.y, expected_slerp.y)
        self.assertAlmostEqual(slerp.z, expected_slerp.z)


class TestLine(unittest.TestCase):
    def test_point_at(self):
        line = Line(Quaternion(0, 1, 1, 1), Quaternion(0, 1, 0, 0))
        result = line.point_at(3)
        self.assertEqual(result, Quaternion(0, 4, 1, 1))

    def test_shortest_distance_to_point(self):
        line = Line(Quaternion(0, 1, 1, 1), Quaternion(0, 1, 0, 0))
        point = Quaternion(0, 4, 4, 4)
        result = line.shortest_distance_to_point(point)
        self.assertAlmostEqual(result, np.sqrt(18))         # chatGPT tested for 27


class TestPlane(unittest.TestCase):
    def test_signed_distance_to_point(self):
        point_on_plane = Quaternion(0, 0, 0, 0)
        plane_normal = Quaternion(0, 0, 1, 0)
        plane = Plane(point_on_plane, plane_normal)

        point = Quaternion(0, 2, 2, 2)
        result = plane.signed_distance_to_point(point)
        self.assertAlmostEqual(result, 2)  # The point is 2 units above the plane in the z-direction

    def test_project_point_onto_plane(self):
        point_on_plane = Quaternion(0, 0, 0, 0)
        plane_normal = Quaternion(0, 0, 0, 1)   # chatGPT exchanged y and z in the test and normal
        plane = Plane(point_on_plane, plane_normal)

        point = Quaternion(0, 2, 2, 2)
        projected_point = plane.project_point_onto_plane(point)

        self.assertAlmostEqual(projected_point.w, 0, "w")  # Real part (w) should be zero
        self.assertAlmostEqual(projected_point.x, 2, "x")  # X-coordinate should remain the same
        self.assertAlmostEqual(projected_point.y, 2, "y")  # Y-coordinate should remain the same
        self.assertAlmostEqual(projected_point.z, 0, "z")  # Z-coordinate should be projected onto the plane (z=0)

    def test_line_plane_intersection(self):
        point_on_line = Quaternion(0, 1, 1, 1)
        line_direction = Quaternion(0, 1, 0, 0)
        line = Line(point_on_line, line_direction)

        point_on_plane = Quaternion(0, 0, 0, 0)
        plane_normal = Quaternion(0, 1, 1, 1)
        plane = Plane(point_on_plane, plane_normal)

        intersection = plane.line_intersection(line)
        self.assertIsNotNone(intersection)

        # Check if the intersection point lies on both the line and the plane
        t = line_direction.dot_product(intersection.subtract(point_on_line)) / line_direction.dot_product(line_direction)
        line_point = line.point_at(t)
        self.assertAlmostEqual(line_point.w, intersection.w, "w")
        self.assertAlmostEqual(line_point.x, intersection.x, "x")
        self.assertAlmostEqual(line_point.y, intersection.y, "y")
        self.assertAlmostEqual(line_point.z, intersection.z, "z")


if __name__ == '__main__':
    unittest.main()
