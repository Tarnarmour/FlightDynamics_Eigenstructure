# rrt dubins path planner for mavsim_python
#
# mavsim_python
#     - Beard & McLain, PUP, 2012
#     - Last updated:
#         4/16/2019 - RWB
import numpy as np
from numpy import sin, cos, sqrt, tan
from numpy.linalg import norm
from message_types.msg_waypoints import MsgWaypoints
from message_types.msg_world_map import MsgWorldMap
from chap11.draw_waypoints import DrawWaypoints
from chap12.draw_map import DrawMap
from chap11.dubins_parameters import DubinsParameters
from parameters.planner_parameters import R_min
import pyqtgraph as pg
import pyqtgraph.opengl as gl

R = np.array([1, 0, 0, 1])
B = np.array([0, 0, 1, 1])
G = np.array([0, 1, 0, 1])
K = np.array([0, 0, 0, 1])
Y = np.array([0.5, 0.5, 0, 1])

class RRTDubins:
    def __init__(self):
        self.segment_length = 200  # standard length of path segments
        self.plot_window = []
        self.plot_app = []
        self.dubins_path = DubinsParameters()
        self.Va = 0.0
        self.radius = 100.0

    def update(self, start_pose, end_pose, Va, world_map=MsgWorldMap(), radius=R_min):

        PLOT = False

        if PLOT:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            w = world_map.building_width
            for b in world_map.building_details:
                pos = b[0:2]
                xs = np.array([-w / 2, w / 2, w / 2, -w / 2, -w / 2]) + pos[0]
                ys = np.array([-w / 2, -w / 2, w / 2, w / 2, -w / 2]) + pos[1]
                ax.plot(xs, ys, color=np.array([0.2, 0.2, 0.2, 1]))

            def scatter(pose, color=B):
                x, y = pose[0], pose[1]
                handle = ax.scatter(x, y, color=color)
                return handle

            def draw_dubins(points, color=B):
                handle, = ax.plot(points[:, 0], points[:, 1], color=color)
                return handle

        self.Va = Va
        self.radius = radius
        # generate tree
        tree = MsgWaypoints()
        tree.type = 'dubins'
        chi_start = start_pose[3]
        start_pose = start_pose[0:3]
        chi_end = end_pose[3]
        end_pose = end_pose[0:3]
        height = end_pose[2]
        # add the start pose to the tree
        tree.add(ned=start_pose, course=chi_start, parent=0, cost=0.0, airspeed=Va)

        if PLOT:
            scatter(start_pose, K)
            scatter(end_pose, K)
        # check to see if start_pose connects directly to end_pose

        while not self.check_finished(tree, end_pose, self.segment_length):
            if np.random.randint(0, 100) > 90:
                p_star = end_pose
            else:
                p_star = random_pose(world_map, pd=height)
            index = self.find_nearest_node(p_star, tree)
            p_last, chi_last = tree.ned[index], tree.course[index]
            chi_star = np.arctan2((p_star - p_last)[1], (p_star - p_last)[0])
            p_star = self.find_valid_node(p_last, p_star, radius)
            self.dubins_path.update(p_last, chi_last, p_star, chi_star, radius)
            points = self.points_along_path()

            if PLOT:
                handle = draw_dubins(points, Y)
                plt.pause(0.001)

            collision = self.collision(world_map)
            if collision:
                if PLOT:
                    handle.remove()
                # print("Collsion detected!")
            else:
                if PLOT:
                    handle.remove()
                    draw_dubins(points, B)
                    scatter(p_star, K)

                cost = tree.cost[index] + 1
                tree.add(ned=p_star, course=chi_star, parent=index, cost=cost, airspeed=Va)

            if PLOT:
                plt.pause(0.01)


        if PLOT:
            plt.close(fig)

        # Add waypoints at the end to extend path
        # tree.add(ned=end_pose * 1.5, course=chi_end, parent=tree.num_waypoints - 1, cost=tree.cost[-1] + 1, airspeed=Va)
        # tree.add(ned=end_pose * 2.0, course=chi_end, parent=tree.num_waypoints - 1, cost=tree.cost[-1] + 1, airspeed=Va)
        # tree.add(ned=end_pose * 2.5, course=chi_end, parent=tree.num_waypoints - 1, cost=tree.cost[-1] + 1, airspeed=Va)

        # find path with minimum cost to end_node
        waypoints_not_smooth = self.find_minimum_path(tree, end_pose)
        waypoints = self.smooth_path(waypoints_not_smooth, world_map, radius)

        return waypoints

    # def extend_tree(self, tree, end_pose, Va, world_map, radius):
    #     # extend tree by randomly selecting pose and extending tree toward that pose
    #     flag =
    #     return flag

    def points_along_path(self, Del=1.0):
        # returns a list of points along the dubins path
        initialize_points = True

        # points along start circle
        q1 = self.dubins_path.p_s - self.dubins_path.center_s
        q2 = self.dubins_path.r1 - self.dubins_path.center_s
        phi1 = np.arctan2(q1[1], q1[0])
        phi2 = np.arctan2(q2[1], q2[0])
        if self.dubins_path.dir_s == -1:
            if phi2 > phi1:
                phis = phi2 - phi1 - 2 * np.pi
            else:
                phis = phi2 - phi1
        elif self.dubins_path.dir_s == 1:
            if phi2 < phi1:
                phis = phi2 - phi1 + 2 * np.pi
            else:
                phis = phi2 - phi1
        else:
            phis = None
        arc_length_s = abs(phis) * self.radius
        if np.isnan(arc_length_s):
            print("error here")
        N_s = int(arc_length_s / Del)

        # loop through first straight line
        v = self.dubins_path.r2 - self.dubins_path.r1
        length_straight = norm(v)
        N_line = int(length_straight / Del)

        # loop through second circle
        q3 = self.dubins_path.r2 - self.dubins_path.center_e
        q4 = self.dubins_path.r3 - self.dubins_path.center_e
        phi3 = np.arctan2(q3[1], q3[0])
        phi4 = np.arctan2(q4[1], q4[0])
        if self.dubins_path.dir_e == -1:
            if phi4 > phi3:
                phie = phi4 - phi3 - 2 * np.pi
            else:
                phie = phi4 - phi3
        elif self.dubins_path.dir_e == 1:
            if phi4 < phi3:
                phie = phi4 - phi3 + 2 * np.pi
            else:
                phie = phi4 - phi3
        else:
            phie = None
            print("error in lame")
        arc_length_e = abs(phie) * self.radius
        N_e = int(arc_length_e / Del)

        N = N_s + N_line + N_e
        points = np.zeros((N, 3))

        for i in range(N_s):
            theta = phi1 + phis * i / N_s
            points[i] = self.dubins_path.center_s + self.radius * rot(theta) @ np.array([1, 0, 0])

        for i in range(N_line):
            points[N_s + i] = self.dubins_path.r1 + v * i / N_line

        for i in range(N_e):
            theta = phi3 + phie * i / N_e
            points[N_line + N_s + i] = self.dubins_path.center_e + self.radius * rot(theta) @ np.array([1, 0, 0])

        return points

    def collision(self, world_map):
        # check to see of path from start_pose to end_pose colliding with world_map
        points = self.points_along_path()
        height_margin = 0.
        width_margin = 0.
        width = world_map.building_width
        for p in points:
            for b in world_map.building_details:
                if p[2] - b[2] < height_margin:
                    v = p[0:2] - b[0:2]
                    if np.max(np.abs(v)) < width_margin + width / 2:
                        print("collision")
                        return True
        return False

    def plot_map(self, world_map, tree, waypoints, smoothed_waypoints, radius):
        scale = 4000
        # initialize Qt gui application and window
        self.plot_app = pg.QtGui.QApplication([])  # initialize QT
        self.plot_window = gl.GLViewWidget()  # initialize the view object
        self.plot_window.setWindowTitle('World Viewer')
        self.plot_window.setGeometry(0, 0, 1500, 1500)  # args: upper_left_x, upper_right_y, width, height
        grid = gl.GLGridItem()  # make a grid to represent the ground
        grid.scale(scale / 20, scale / 20, scale / 20)  # set the size of the grid (distance between each line)
        self.plot_window.addItem(grid)  # add grid to viewer
        self.plot_window.setCameraPosition(distance=scale, elevation=50, azimuth=-90)
        self.plot_window.setBackgroundColor('k')  # set background color to black
        self.plot_window.show()  # display configured window
        self.plot_window.raise_()  # bring window to the front

        blue = np.array([[30, 144, 255, 255]]) / 255.
        red = np.array([[204, 0, 0]]) / 255.
        green = np.array([[0, 153, 51]]) / 255.
        DrawMap(world_map, self.plot_window)
        DrawWaypoints(waypoints, radius, blue, self.plot_window)
        DrawWaypoints(smoothed_waypoints, radius, red, self.plot_window)
        self.draw_tree(radius, green)
        # draw things to the screen
        self.plot_app.processEvents()

    def draw_tree(self, radius, color, window, tree=None):
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        Del = 0.05
        for i in range(1, tree.num_waypoints):
            parent = int(tree.parent.item(i))
            self.dubins_path.update(column(tree.ned, parent), tree.course[parent],
                                    column(tree.ned, i), tree.course[i], radius)
            points = self.points_along_path(Del)
            points = points @ R.T
            tree_color = np.tile(color, (points.shape[0], 1))
            tree_plot_object = gl.GLLinePlotItem(pos=points,
                                                 color=tree_color,
                                                 width=2,
                                                 antialias=True,
                                                 mode='line_strip')
            self.plot_window.addItem(tree_plot_object)

    def find_minimum_path(self, tree, end_pose):
        # find the lowest cost path to the end node
        p_last = tree.ned[-1]
        chi_last = tree.course[-1]
        cost_prev = tree.cost[-1]
        parent = tree.num_waypoints - 1

        v = end_pose - p_last
        p_star = p_last + v / norm(v) * max(norm(v), 4.1 * self.radius)
        chi_star = np.arctan2(v[1], v[0])
        tree.add(ned=p_star, course=chi_star, parent=parent, cost=cost_prev + 1, airspeed=self.Va)
        # find nodes that connect to end_node
        connecting_nodes = [tree.num_waypoints - 1]

        # find minimum cost last node
        idx = connecting_nodes[0]
        while idx > 0:
            idx = tree.parent[idx]
            connecting_nodes.append(idx)

        # construct lowest cost path order
        connecting_nodes.reverse()
        # construct waypoint path
        waypoints = MsgWaypoints()
        for i in connecting_nodes:
            p = tree.ned[i]
            course = tree.course[i]
            airspeed = tree.airspeed[i]
            waypoints.add(ned=p, course=course, airspeed=airspeed)

        return waypoints

    def smooth_path(self, waypoints, world_map, radius):
        # smooth the waypoint path
        smooth = [0]  # add the first waypoint

        p_cur, chi_cur = waypoints.ned[0], waypoints.course[0]
        for i in range(1, waypoints.num_waypoints):
            p_next = waypoints.ned[i]
            chi_next = waypoints.course[i]
            self.dubins_path.update(p_cur, chi_cur, p_next, chi_next, self.radius)

            if self.collision(world_map) or i == waypoints.num_waypoints - 1:
                smooth.append(i - 1)
                p_cur, chi_cur = waypoints.ned[i - 1], waypoints.course[i - 1]

        # construct smooth waypoint path
        smooth_waypoints = MsgWaypoints()
        for i in smooth:
            p = waypoints.ned[i]
            course = waypoints.course[i]
            airspeed = waypoints.airspeed[i]
            smooth_waypoints.add(ned=p, course=course, airspeed=airspeed)

        return smooth_waypoints

    def find_nearest_node(self, p_star, tree=MsgWaypoints()):
        dist = np.inf
        index = 0
        for i, p in enumerate(tree.ned):
            D = abs(norm(p - p_star) - 3*R_min)
            if D < dist:
                index = i
                dist = D
        return index

    def find_valid_node(self, p1, p2, radius=R_min):
        v = p2 - p1
        d = norm(v)
        q = v / d
        L = max(self.segment_length, 4.1 * radius)
        p_star = p1 + q * L
        # if d < 3 * radius:
        #     p_star = p1 + q * 3.1 * radius
        # else:
        #     p_star = p2
        return p_star

    def check_finished(self, tree, end_pose, max_length):
        p_last, p_end = tree.ned[-1], end_pose
        if norm(p_last - p_end) < max_length:
            return True
        else:
            return False

# def distance(start_pose, end_pose):
#     # compute distance between start and end pose
#     d =
#     return d


# def height_above_ground(world_map, point):
#     # find the altitude of point above ground level
#     point_height =
#     if (d_n < world_map.building_width) and (d_e < world_map.building_width):
#         map_height =
#     else:
#         map_height =
#     h_agl =
#     return h_agl


def random_pose(world_map, pd):
    # generate a random pose
    pose = np.random.random_sample((3,)) * world_map.city_width * 1.5 - 0.0 * world_map.city_width
    pose[2] = pd
    course = np.random.random_sample((1,)) * 2 * np.pi - np.pi
    return pose  # , course[0]


def mod(x):
    # force x to be between 0 and 2*pi
    while x < 0:
        x += 2 * np.pi
    while x > 2 * np.pi:
        x -= 2 * np.pi
    return x


def column(A, i):
    # extracts the ith column of A and return column vector
    tmp = A[:, i]
    col = tmp.reshape(A.shape[0], 1)
    return col


def rot(th):
    return np.array([[cos(th), -sin(th), 0],
                     [sin(th), cos(th), 0],
                     [0, 0, 1]])