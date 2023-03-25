import unittest
import numpy as np
from quaternion import Quaternion, Line, Plane

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
        plane = Plane(Quaternion(0, 1, 1, 1), Quaternion(0, 1, 1, 1))
        point = Quaternion(0, 2, 2, 2)
        result = plane.signed_distance_to_point(point)
        self.assertAlmostEqual(result, 1 / np.sqrt(3))

    def test_project_point_onto_plane(self):
        plane = Plane(Quaternion(0, 1, 1, 1), Quaternion(0, 1, 1, 1))
        point = Quaternion(0, 2, 2, 2)
        result = plane.project_point_onto_plane(point)
        expected_result = Quaternion(0, 4/3, 4/3, 4/3)
        self.assertAlmostEqual(result.w, expected_result.w)
        self.assertAlmostEqual(result.x, expected_result.x)
        self.assertAlmostEqual(result.y, expected_result.y)
        self.assertAlmostEqual(result.z, expected_result.z)

if __name__ == '__main__':
    unittest.main()
