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
        q1 = Quaternion(0,1, 2, 3)
        q2 = Quaternion(0, 4, 5, 6)
        result = q1.cross_product(q2)
        self.assertEqual(result, Quaternion(0, -3, 6, -3))

class TestLine(unittest.TestCase):
    def test_point_at(self):
        line = Line(Quaternion(0, 1, 1, 1), Quaternion(0, 1, 0, 0))
        result = line.point_at(3)
        self.assertEqual(result, Quaternion(0, 4, 1, 1))

    def test_shortest_distance_to_point(self):
        line = Line(Quaternion(0, 1, 1, 1), Quaternion(0, 1, 0, 0))
        point = Quaternion(0, 4, 4, 4)
        result = line.shortest_distance_to_point(point)
        self.assertAlmostEqual(result, np.sqrt(27))


class TestPlane(unittest.TestCase):
    def test_signed_distance_to_point(self):
        point_on_plane = Quaternion(0, 0, 0, 0)
        plane_normal = Quaternion(0, 0, 1, 0)
        plane = Plane(point_on_plane, plane_normal)

        point = Quaternion(0, 2, 2, 2)
        result = plane.signed_distance_to_point(point)
        self.assertAlmostEqual(result, 2)  # The point is 2 units above the plane in the z-direction

    def test_project_point_onto_plane(self):
        plane = Plane(Quaternion(0, 1, 1, 1), Quaternion(0, 1, 1, 1))
        point = Quaternion(0, 2, 2, 2)
        result = plane.project_point_onto_plane(point)
        expected_result = Quaternion(0, 4/3, 4/3, 4/3)
        self.assertAlmostEqual(result.w, expected_result.w)
        self.assertAlmostEqual(result.x, expected_result.x)
        self.assertAlmostEqual(result.y, expected_result.y)
        self.assertAlmostEqual(result.z, expected_result.z)

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
        self.assertAlmostEqual(line_point.w, intersection.w)
        self.assertAlmostEqual(line_point.x, intersection.x)
        self.assertAlmostEqual(line_point.y, intersection.y)
        self.assertAlmostEqual(line_point.z, intersection.z)


if __name__ == '__main__':
    unittest.main()
