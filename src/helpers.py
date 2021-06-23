from math import sqrt, cos, sin, pi
import numpy as np
import transforms3d


def transform_world_to_ego_frame(pos, ego, ego_rpy):
    """
    transforms a position expressed in world frame into vehicle frame. 

    Args:
        pos (x,y,z): Position to transform
        ego (x, y, yaw, current_speed): state of the vehicle
        ego_rpy (current_roll, current_pitch, current_yaw): current rpy of the vehicle

    Returns:
        loc_relative: position expressed in vehicle frame
    """
    loc = np.array(pos) - np.array(ego)
    r = transforms3d.euler.euler2mat(ego_rpy[0], ego_rpy[1], ego_rpy[2]).T
    loc_relative = np.dot(r, loc)
    return loc_relative


# Using d = (v_f^2 - v_i^2) / (2 * a), compute the distance
# required for a given acceleration/deceleration.
def calc_distance(v_i, v_f, a):
    """Computes the distance given an initial and final speed, with a constant
    acceleration.

    args:
        v_i: initial speed (m/s)
        v_f: final speed (m/s)
        a: acceleration (m/s^2)
    returns:
        d: the final distance (m)
    """
    return (v_f * v_f - v_i * v_i) / 2 / a


# Using v_f = sqrt(v_i^2 + 2ad), compute the final speed for a given
# acceleration across a given distance, with initial speed v_i.
# Make sure to check the discriminant of the radical. If it is negative,
# return zero as the final speed.
def calc_final_speed(v_i, a, d):
    """Computes the final speed given an initial speed, distance travelled,
    and a constant acceleration.

    args:
        v_i: initial speed (m/s)
        a: acceleration (m/s^2)
        d: distance to be travelled (m)
    returns:
        v_f: the final speed (m/s)
    """
    pass

    temp = v_i * v_i + 2 * d * a
    if temp < 0:
        return 0.0000001
    else:
        return sqrt(temp)


def filter_bbs_by_depth(boxes, distance_image):
    """
    Returns a list of BoundBox objects filtered by their mean distance.
    distance_image is an array of distance measures in meters with the shape of the depth image.
    """
    new_boxes = []
    img_shape = distance_image.shape  # assumed (h, w, color)
    if boxes:
        for box in boxes:
            bounds = box.get_bounds()  # (x_min, y_min, x_max, y_max)
            bounds = (int(bounds[0] * img_shape[1]),
                      int(bounds[1] * img_shape[0]),
                      int(bounds[2] * img_shape[1]),
                      int(bounds[3] * img_shape[0]))
            bounded_dist = distance_image[bounds[1]:bounds[3], bounds[0]:bounds[2]]

            # Gaussian-like kernel
            x, y = np.meshgrid(np.linspace(-1, 1, bounded_dist.shape[1]), np.linspace(-1, 1, bounded_dist.shape[0]))
            d = np.sqrt(x * x + y * y)
            sigma, mu = 1.0, 0.0
            g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

            dist = np.average(bounded_dist * g)
            # dist = bounded_dist[int(bounded_dist.shape[0]/2), int(bounded_dist.shape[1]/2)]
            # dist = np.average(bounded_dist)
            if dist < 1_000:
                new_boxes.append(box)

    return new_boxes


def zoom_image(image, zoom_factor):
    """
    Returns the image zoomed towards the center by the zoom factor.
    """
    image = image.copy()
    img_size = (int(image.shape[0] / zoom_factor), int(image.shape[1] / zoom_factor))
    img_center = (int(image.shape[0] / 2), int(image.shape[1] / 2))

    image = image[int(img_center[0] - img_size[0] / 2):int(img_center[0] + img_size[0] / 2),
                  int(img_center[1] - img_size[1] / 2):int(img_center[1] + img_size[0] / 2)]

    return image


def line_intersection(line1, line2):
    """
    Returns True if line1 and line2 intersect, False otherwise
    """
    def ccw(x, y, z):
        return (z[1] - x[1]) * (y[0] - x[0]) > (y[1] - x[1]) * (z[0] - x[0])

    a = line1[0]
    b = line1[1]
    c = line2[0]
    d = line2[1]
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def rotate_x(angle):
    r = np.mat([[1, 0, 0],
                [0, cos(angle), -sin(angle)],
                [0, sin(angle), cos(angle)]])
    return r


def rotate_y(angle):
    r = np.mat([[cos(angle), 0, sin(angle)],
                [0, 1, 0],
                [-sin(angle), 0, cos(angle)]])
    return r


def rotate_z(angle):
    r = np.mat([[cos(angle), -sin(angle), 0],
                [sin(angle), cos(angle), 0],
                [0, 0, 1]])
    return r


# Transform the obstacle with its boundary point in the global frame
def get_obstacle_box_points(location, dimensions, rotation, x_shift=0):
    box_pts = []

    x = location.x
    y = location.y
    z = location.z

    yaw = rotation.yaw * pi / 180

    x_rad = dimensions.x + x_shift
    y_rad = dimensions.y
    z_rad = dimensions.z

    # Border points in the obstacle frame
    pos = np.array([[-x_rad, -x_rad, -x_rad, 0, x_rad, x_rad, x_rad, 0],
                    [-y_rad, 0, y_rad, y_rad, y_rad, 0, -y_rad, -y_rad]])

    # Rotation of the obstacle
    rot_yam = np.array([[np.cos(yaw), np.sin(yaw)],
                        [-np.sin(yaw), np.cos(yaw)]])

    # Location of the obstacle in the world frame
    pos_shift = np.array([[x, x, x, x, x, x, x, x],
                          [y, y, y, y, y, y, y, y]])

    pos = np.add(np.matmul(rot_yam, pos), pos_shift)

    for j in range(pos.shape[1]):
        box_pts.append([pos[0, j], pos[1, j]])

    return box_pts


def transform_to_matrix(transform):
    rotation = transform.rotation
    rotation = np.deg2rad([rotation.roll, rotation.pitch, rotation.yaw])
    location = transform.location
    rotation_matrix = transforms3d.euler.euler2mat(rotation[0], rotation[1], rotation[2])
    matrix = np.append(rotation_matrix, [[location.x], [location.y], [location.z]], axis=1)
    matrix = np.vstack([matrix, [0, 0, 0, 1]])
    return matrix


def estimate_next_entity_pos(entity, speed=None, speed_scale_factor=1):
    """
    Returns global frame coordinates of next position for entity.
    speed is the entity speed (by default it is .forward_speed).
    speed_scale_factor is used for simulation step
    """
    if speed is None:
        speed = entity.forward_speed
    return translate_position(entity, speed * speed_scale_factor)


def translate_position(entity, offset):
    """
    Returns global frame coordinates of the position for entity translated by offset on the x-axis
    """
    return (transform_to_matrix(entity.transform) @ np.array([offset, 0, 0, 1]))[:-1]


def sad(a, b):
    """
    Given two angles returns the difference in the range [-180, 180]:

    Parameters
    ----------
    a : float
        The first angle
    b : float
        The second angle

    Return
    ----------
    d : float
        The difference in the range [-180, 180]

    """
    d = b - a
    if d > 180:
        d -= 360
    if d < -180:
        d += 360
    return d


def check_obstacle_future_intersection(entity, dist, loc_relative, ego, ego_rpy, speed=None):
    """
    Given the entity, the relative location and the ego_state checks for any intersections.
    """
    # Check if entity's trajectory intersect ego's trajectory
    a = (0, 0)
    b = (dist,0)
    c = transform_world_to_ego_frame(
        translate_position(entity, -5),
        ego,
        ego_rpy
    )[:-1]
    d = transform_world_to_ego_frame(
        estimate_next_entity_pos(entity, speed = speed),
        ego,
        ego_rpy
    )[:-1]

    return line_intersection((a, b), (c, d))
