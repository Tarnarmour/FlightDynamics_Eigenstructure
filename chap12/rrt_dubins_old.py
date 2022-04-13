# rrt dubins path planner for mavsim_python
#
# mavsim_python
#     - Beard & McLain, PUP, 2012
#     - Last updated:
#         4/16/2019 - RWB
import numpy as np
from numpy.linalg import norm
from numpy import sin, cos, tan, sqrt
from message_types.msg_waypoints import MsgWaypoints
from message_types.msg_world_map import MsgWorldMap
from parameters.planner_parameters import R_min
from chap11.draw_waypoints import DrawWaypoints
from chap12.draw_map import DrawMap
from chap11.dubins_parameters import DubinsParameters
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt

R = np.array([1, 0, 0, 1])
B = np.array([0, 0, 1, 1])
G = np.array([0, 1, 0, 1])
K = np.array([0, 0, 0, 1])


class RRTDubins:
    def __init__(self):
        self.segment_length = 300.0  # standard length of path segments
        self.plot_window = []
        self.plot_app = []
        self.dubins_path = DubinsParameters()
        self.ax = None
        self.fig = None

    def update(self, start_pose, end_pose, Va, world_map, radius):
        self.fig, self.ax = plt.subplots()
        scatter(self.ax, start_pose, K)
        scatter(self.ax, end_pose, G)

        for d in world_map.building_details:
            draw_building(self.ax, d[0:2], w=world_map.building_width)
        #generate tree
        tree = MsgWaypoints()
        tree.type = 'dubins'
        # add the start pose to the tree
        tree.add(start_pose[0:3], airspeed=Va, course=start_pose[3], cost=0, parent=-1)
        # check to see if start_pose connects directly to end_pose
        connected = False
        while not connected:
            connected = self.extend_tree(tree, end_pose, Va, world_map, radius)
        tree.add(end_pose[0:3], airspeed=Va, course=end_pose[3])
        # find path with minimum cost to end_node
        waypoints_not_smooth = find_minimum_path(tree)
        waypoints = smooth_path(tree, world_map)

        return waypoints

    def extend_tree(self, tree, end_pose, Va, world_map, radius):
        # extend tree by randomly selecting pose and extending tree toward that pose
        flag = False
        p_star = random_pose(world_map, 100)
        index = find_closest_node(tree, p_star)
        p_prev = tree.ned[index]
        course_prev = tree.course[index]
        q = p_star[0:3] - p_prev
        d = norm(q)
        p_new = p_prev + q / d * min(self.segment_length, d)
        course_new = np.arctan2(q[1], q[0])

        if not collision(p_prev, p_new, course_prev, course_new, world_map, radius, self.ax):
            tree.add(ned=p_new, parent=index, airspeed=Va, course=course_new)
            flag = norm(end_pose[0:3] - p_new) < self.segment_length

        return flag

    def plot_map(self, world_map, tree, waypoints, smoothed_waypoints, radius):
        scale = 4000
        # initialize Qt gui application and window
        self.plot_app = pg.QtGui.QApplication([])  # initialize QT
        self.plot_window = gl.GLViewWidget()  # initialize the view object
        self.plot_window.setWindowTitle('World Viewer')
        self.plot_window.setGeometry(0, 0, 1500, 1500)  # args: upper_left_x, upper_right_y, width, height
        grid = gl.GLGridItem() # make a grid to represent the ground
        grid.scale(scale/20, scale/20, scale/20) # set the size of the grid (distance between each line)
        self.plot_window.addItem(grid) # add grid to viewer
        self.plot_window.setCameraPosition(distance=scale, elevation=50, azimuth=-90)
        self.plot_window.setBackgroundColor('k')  # set background color to black
        self.plot_window.show()  # display configured window
        self.plot_window.raise_() # bring window to the front

        blue = np.array([[30, 144, 255, 255]])/255.
        red = np.array([[204, 0, 0]])/255.
        green = np.array([[0, 153, 51]])/255.
        DrawMap(world_map, self.plot_window)
        DrawWaypoints(waypoints, radius, blue, self.plot_window)
        DrawWaypoints(smoothed_waypoints, radius, red, self.plot_window)
        self.draw_tree(radius, green)
        # draw things to the screen
        self.plot_app.processEvents()

    def draw_tree(self, tree, radius, color):
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        Del = 0.05
        for i in range(1, tree.num_waypoints):
            parent = int(tree.parent.item(i))
            self.dubins_path.update(column(tree.ned, parent), tree.course[parent],
                                    column(tree.ned, i), tree.course[i], radius)
            points = points_along_path(Del)
            points = points @ R.T
            tree_color = np.tile(color, (points.shape[0], 1))
            tree_plot_object = gl.GLLinePlotItem(pos=points,
                                                color=tree_color,
                                                width=2,
                                                antialias=True,
                                                mode='line_strip')
            self.plot_window.addItem(tree_plot_object)


def points_along_path(ps, pe, chis, chie, radius, Del=5):
    # returns a list of points along the dubins path
    dubins_path = DubinsParameters(ps, chis, pe, chie, radius)
    # dubins_path.update(ps, chis, pe, chie, radius)
    # points along start circle
    # Find the amount of rotation about the first circle

    q1s = dubins_path.p_s - dubins_path.center_s
    q2s = dubins_path.r1 - dubins_path.center_s
    th = np.arctan2(q1s[1], q1s[0])
    R = rot(th)
    e2 = R.T @ - q2s
    phi_s = np.arctan2(e2[1], e2[0])
    phi0_s = np.arctan2(q1s[1], q1s[0])
    # if np.sign(th * dubins_path.dir_s) > 0:
    #     phi_s = phi_s - 2 * np.pi

    arc_length = abs(phi_s) * radius
    N_cs = int(arc_length / Del)
    v = dubins_path.r2 - dubins_path.r1
    straight_length = norm(v)
    N_s = int(straight_length / Del)

    q1e = dubins_path.r2 - dubins_path.center_e
    q2e = dubins_path.r3 - dubins_path.center_e
    th = np.arctan2(q1e[1], q1e[0])
    R = rot(th)
    e2 = R.T @ -q2e
    phi_e = np.arctan2(e2[1], e2[0])
    phi0_e = np.arctan2(q1e[1], q1e[0])
    # if np.sign(th * dubins_path.dir_e) < 0:
    #     phi_e = phi_e - 2 * np.pi
    arc_length = abs(phi_e) * radius
    N_ce = int(arc_length / Del)

    theta_list = np.linspace(0, phi_s, N_cs) + phi0_s

    points = np.zeros((N_cs + N_s + N_ce, 3))
    for i, ang in enumerate(theta_list):
        points[i, :] = dubins_path.center_s + radius * rot(ang) @ np.array([1, 0, 0])

    # points along straight line
    lengths = np.linspace(0, 1, N_s)
    for i, l in enumerate(lengths):
        points[N_cs + i, :] = dubins_path.r1 + v * l

    # points along end circle
    theta_list = np.linspace(0, phi_e, N_ce) + phi0_e

    for i, ang in enumerate(theta_list):
        points[N_cs + N_s + i, :] = dubins_path.center_e + radius * rot(ang) @ np.array([1, 0, 0])
    return points


def point_valid(points, world_map):
    height_margin = 20
    width_margin = 20
    for p in points:
        for b in world_map.building_details:
            if p[2] - b[2] < height_margin:
                return True
            v = p[0:2] - b[0:2]
            if abs(v[0]) < width_margin and abs(v[1]) < width_margin:
                return True
    return False


def collision(p_prev, p_new, course_prev, course_new, world_map, radius=R_min, ax=None):
    points = points_along_path(p_prev, p_new, course_prev, course_new, radius)
    ax.plot(points[:, 0], points[:, 1], color=R)
    plt.pause(0.1)
    collision = point_valid(points, world_map)
    return collision


def find_closest_node(tree, p_star):
    dist = np.inf
    index = 0
    for i, n in enumerate(tree.ned):
        D = norm(n[0:3] - p_star[0:3])
        if D < dist:
            dist = D
            index = i
    return index


def find_minimum_path(tree, end_pose):
    # find the lowest cost path to the end node
    # find nodes that connect to end_node
    connecting_nodes = [tree.num_waypoints - 1]
    reached_start = False
    while not reached_start:
        connecting_nodes.append(tree.parent[connecting_nodes[-1]])
        if connecting_nodes[-1] == 0:
            reached_start = True

    # construct waypoint path
    waypoints = MsgWaypoints()
    for i in range(len(connecting_nodes)):
        idx = connecting_nodes[len(connecting_nodes) - 1 - i]
        p_new = tree.ned[idx]
        waypoints.add(ned=p_new, airspeed=tree.airspeed[idx], course=tree.course[idx])

    return waypoints


def smooth_path(waypoints, world_map, radius):
    smooth = [0]  # add the first waypoint

    p_cur = waypoints.ned[0]
    for i, p_next in enumerate(waypoints.ned):

        if collision(p_cur, p_next, world_map):
            smooth.append(i - 1)
            p_cur = waypoints.ned[i - 1]

    # construct smooth waypoint path
    smooth_waypoints = MsgWaypoints()
    for i in smooth:
        if i == len(smooth) - 1:
            course = 0.0
        else:
            ned_next = waypoints.ned[i+1]
            course = np.arctan2(ned_next[1], ned_next[0])
        smooth_waypoints.add(ned=waypoints.ned[i], airspeed=waypoints.airspeed[i],
                             course=course)

    # append extra to the end to make line follower happy
    smooth_waypoints.add(ned=waypoints.ned[-1] * 1.1, airspeed=waypoints.airspeed[-1],
                         course=waypoints.course[-1])

    smooth_waypoints.add(ned=waypoints.ned[-1] * 1.1, airspeed=waypoints.airspeed[-1],
                         course=waypoints.course[-1])

    return smooth_waypoints


def distance(start_pose, end_pose):
    # compute distance between start and end pose
    d = norm(start_pose[0:3] - end_pose[0:3])
    return d


# def height_above_ground(world_map, point):
#     # find the altitude of point above ground level
#     point_height =
#     if (d_n<world_map.building_width) and (d_e<world_map.building_width):
#         map_height =
#     else:
#         map_height =
#     h_agl =
#     return h_agl


def random_pose(world_map, pd):
    # generate a random pose
    pose = np.random.random_sample((4,))
    pose[0:2] = pose[0:2] * world_map.city_width
    pose[2] = -pd  # height_above_ground(world_map, pose[0:2])
    pose[3] = 0.0  # wait to set course
    return pose


def mod(x):
    # force x to be between 0 and 2*pi
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x


def scatter(ax, pose, color=np.array([0, 0, 1, 1])):
    handle = ax.scatter(pose[0], pose[1], color=color)
    return handle


def plot(ax, pose1, pose2, color=np.array([0, 0, 1, 1]), ls='-'):
    handle, = ax.plot([pose1[0], pose2[0]], [pose1[1], pose2[1]], color=color, ls=ls)
    return handle


def draw_building(ax=plt.axis(), pos=np.array([0, 0]), w=20, color=np.array([0.2, 0.2, 0.2, 1])):
    xs = np.array([-w / 2, w / 2, w / 2, -w / 2, -w / 2]) + pos[0]
    ys = np.array([-w / 2, -w / 2, w / 2, w / 2, -w / 2]) + pos[1]
    ax.plot(xs, ys, color=color)

def column(A, i):
    # extracts the ith column of A and return column vector
    tmp = A[:, i]
    col = tmp.reshape(A.shape[0], 1)
    return col

def rot(th):
    return np.array([[cos(th), -sin(th), 0],
                     [sin(th), cos(th), 0],
                     [0, 0, 1]])