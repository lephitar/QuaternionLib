import json
import numpy as np
from typing import TypeVar, Union

L = TypeVar("L", bound="Line")
P = TypeVar("P", bound="Plane")

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.w}, {self.x}, {self.y}, {self.z})"

    def from_vector(vec):
        return Quaternion(0, vec[0], vec[1], vec[2])

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def normalize(self):
        norm = self.magnitude()
        return Quaternion(self.w/norm, self.x/norm, self.y/norm, self.z/norm)

    def multiply(self, other):
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return Quaternion(w, x, y, z)

    def add(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def subtract(self, other):
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def scalar_multiply(self, scalar):
        return Quaternion(self.w * scalar, self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def rotate(self, vec):
        vec_quat = Quaternion.from_vector(vec)
        rotated_vec = self.multiply(vec_quat).multiply(self.conjugate())
        return np.array([rotated_vec.x, rotated_vec.y, rotated_vec.z])

    def dot_product(self, other):
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

    def cross_product(self, other):
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return Quaternion(0, x, y, z)

    def slerp(q1, q2, t):
        cos_half_theta = q1.dot_product(q2)
        if abs(cos_half_theta) >= 1.0:
            return q1

        half_theta = np.arccos(cos_half_theta)
        sin_half_theta = np.sqrt(1.0 - cos_half_theta**2)

        if abs(sin_half_theta) < 0.001:
            return Quaternion(q1.w * 0.5 + q2.w * 0.5,
                              q1.x * 0.5 + q2.x * 0.5,
                              q1.y * 0.5 + q2.y * 0.5,
                              q1.z * 0.5 + q2.z * 0.5)

        ratio_a = np.sin((1 - t) * half_theta) / sin_half_theta
        ratio_b = np.sin(t * half_theta) / sin_half_theta

        return Quaternion(q1.w * ratio_a + q2.w * ratio_b, q1.x * ratio_a + q2.x * ratio_b,q1.y * ratio_a + q2.y * ratio_b, q1.z * ratio_a + q2.z * ratio_b)

    def angle_between_quaternions(q1, q2) -> float:
        dot_product = q1.dot_product(q2)
        angle = 2 * np.arccos(np.clip(dot_product, -1.0, 1.0))
        return angle

    def from_axis_angle(axis: [float], angle: float) -> 'Quaternion':
        norm = np.sqrt(sum([x**2 for x in axis]))
        s = np.sin(angle/2) / norm
        c = np.cos(angle/2)
        return Quaternion(c, axis[0]*s, axis[1]*s, axis[2]*s)

    def __eq__(self, other) -> bool:
        return self.w == other.w and \
               self.x == other.x and \
               self.y == other.y and \
               self.z == other.z


def read_json_points(file_path):
    with open(file_path, "r") as file:
        point_data = json.load(file)
    return point_data

def write_json_points(file_path, point_data):
    with open(file_path, "w") as file:
        json.dump(point_data, file, indent=4)

def main():
    # Read points from JSON file
    input_file = "input_points.json"
    point_data = read_json_points(input_file)
    processed_points = []

    rotation_quaternion = Quaternion(0, 1, 0, 0)  # Example quaternion for rotation

    for point in point_data:
        input_vec = np.array([point['x'], point['y'], point['z']])
        rotated_vec = rotation_quaternion.rotate(input_vec)
        processed_points.append({
            'x': rotated_vec[0],
            'y': rotated_vec[1],
            'z': rotated_vec[2]
        })

    # Write processed points to JSON file
    output_file = "output_points.json"
    write_json_points(output_file, processed_points)

if __name__ == "__main__":
    main()



class Line:
    def __init__(self, point: Quaternion, direction: Quaternion):
        self.point = point
        self.direction = direction.normalize()

    def point_at(self, t: float):
        return self.point.add(self.direction.scalar_multiply(t))

    def shortest_distance_to_point(self, point: Quaternion):
        direction_point_to_line = self.direction.cross_product(point.subtract(self.point))
        return direction_point_to_line.magnitude() / self.direction.magnitude()

    def closest_point_on_line_to_point(line: L, point: Quaternion):
        direction_point_to_line = line.direction.cross_product(point.subtract(line.point))
        projected_direction = direction_point_to_line.cross_product(line.direction)
        return line.point.add(projected_direction)


class Plane:
    def __init__(self, point: Quaternion, normal: Quaternion):
        self.point = point
        self.normal = normal.normalize()

    def signed_distance_to_point(self, point: Quaternion):
        return self.normal.dot_product(point.subtract(self.point))

    def project_point_onto_plane(self, point: Quaternion):
        signed_distance = self.signed_distance_to_point(point)
        projection_vector = self.normal.scalar_multiply(signed_distance)
        return point.subtract(projection_vector)

    def reflect_point_across_plane(point: Quaternion, plane: P):
        signed_distance = plane.signed_distance_to_point(point)
        reflection_vector = plane.normal.scalar_multiply(2 * signed_distance)
        return point.subtract(reflection_vector)

    def line_intersection(self, line: L) -> Union[Quaternion, None]:
        line_direction_dot_normal = line.direction.dot_product(self.normal)
        if abs(line_direction_dot_normal) < 1e-6:  # Line is parallel to the plane
            return None
        t = self.signed_distance_to_point(line.point) / line_direction_dot_normal
        intersection_point = line.point_at(-t)
        return intersection_point
