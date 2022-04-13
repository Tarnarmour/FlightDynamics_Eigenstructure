# rrt straight line path planner for mavsim_python
#
# mavsim_python
#     - Beard & McLain, PUP, 2012
#     - Last updated:
#         4/3/2019 - Brady Moon
#         4/11/2019 - RWB
#         3/31/2020 - RWB
import numpy as np
from numpy.linalg import norm
from numpy import sin, cos, tan, sqrt
from message_types.msg_waypoints import MsgWaypoints
from message_types.msg_world_map import MsgWorldMap
from chap11.draw_waypoints import DrawWaypoints
from chap12.draw_map import DrawMap
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt

R = np.array([1, 0, 0, 1])
B = np.array([0, 0, 1, 1])
G = np.array([0, 1, 0, 1])
K = np.array([0, 0, 0, 1])


class RRTStraightLine:
    def __init__(self):
        self.segment_length = 200.0 # standard length of path segments
        self.plot_window = []
        self.plot_app = []

    def update(self, start_pose, end_pose, Va, world_map=MsgWorldMap(), radius=150.0):
        start_pose = np.copy(start_pose[0:3])
        end_pose = np.copy(end_pose[0:3])
        # fig, ax = plt.subplots()
        # scatter(ax, start_pose, K)
        # scatter(ax, end_pose, G)

        # for d in world_map.building_details:
        #     draw_building(ax, d[0:2], w=world_map.building_width)

        #generate tree
        tree = MsgWaypoints()
        #tree.type = 'straight_line'
        tree.type = 'fillet'
        # add the start pose to the tree
        tree.add(ned=start_pose, cost=0, parent=0, connect_to_goal=0)
        path_finished = False
        count = 0
        while not path_finished:
            if count == 3:
                new_pose_type = 'end'
                count = 0
            else:
                new_pose_type = 'random'

            path_finished = self.extend_tree(tree, end_pose, Va, world_map, None, new_pose_type=new_pose_type)
            count += 1
            print(count)
            # plt.pause(0.02)

        cost_new = get_new_cost(end_pose, end_pose) + tree.cost[tree.parent[-1]]
        parent = tree.parent[-1]
        tree.add(ned=end_pose, cost=cost_new, parent=parent, airspeed=Va)

        # plt.close(fig)

        # find path with minimum cost to end_node
        waypoints_not_smooth = find_minimum_path(tree, end_pose)
        waypoints = smooth_path(waypoints_not_smooth, world_map)
        return waypoints

    def extend_tree(self, tree, end_pose, Va, world_map, ax=None, new_pose_type='random'):
        # extend tree by randomly selecting pose and extending tree toward that pose

        if new_pose_type == 'random':
            p_star = random_pose(world_map, 100)
        elif new_pose_type == 'end':
            p_star = end_pose

        indices = find_closest_node(tree, p_star)
        index = indices[0]
        p_prev = tree.ned[index]
        q = p_star - p_prev
        d = norm(q)
        p_new = p_prev + q / d * min(self.segment_length, d)
        # handle = scatter(ax, p_new, color=np.array([0.5, 0.5, 0, 1]))

        # if collision(p_prev, p_new, world_map):
        #     handle.remove()
        #     index = indices[1]
        #     p_prev = tree.ned[index]
        #     q = p_star - p_prev
        #     d = norm(q)
        #     v_star = p_prev + q / d * min(self.segment_length, d)
        #     handle = scatter(ax, v_star, color=np.array([0.5, 0.5, 0, 1]))

        # plt.pause(0.02)
        if not collision(p_prev, p_new, world_map):
            cost_new = get_new_cost(p_new, end_pose) + tree.cost[index]
            parent = index
            tree.add(ned=p_new, cost=cost_new, parent=parent, airspeed=Va)
            flag = norm(end_pose - p_new) < self.segment_length

            # scatter(ax, p_new, B)
            # plot(ax, p_new, p_prev, B, '--')
        else:
            flag = False
        # handle.remove()
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
        #self.plot_window.raise_() # bring window to the front

        blue = np.array([[30, 144, 255, 255]])/255.
        red = np.array([[204, 0, 0]])/255.
        green = np.array([[0, 153, 51]])/255.
        DrawMap(world_map, self.plot_window)
        DrawWaypoints(waypoints, radius, blue, self.plot_window)
        DrawWaypoints(smoothed_waypoints, radius, red, self.plot_window)
        draw_tree(tree, green, self.plot_window)
        # draw things to the screen
        self.plot_app.processEvents()


def get_new_cost(p_new, end_pose):
    cost = 1.0
    return cost


def find_closest_node(tree, p):
    # Find the node in tree which is closest to new configuration p, and return the index
    # of that node
    index = [0, 0]
    dist = np.inf
    for i, node in enumerate(tree.ned):
        dist_new = distance(p, node)
        if dist_new < dist:
            dist = dist_new
            index[1] = index[0]
            index[0] = i
    return index


def smooth_path(waypoints, world_map):
    # smooth the waypoint path
    smooth = [0]  # add the first waypoint

    p_cur = waypoints.ned[0]
    for i, p_next in enumerate(waypoints.ned):

        if collision(p_cur, p_next, world_map):
            smooth.append(i-1)
            p_cur = waypoints.ned[i-1]

    # construct smooth waypoint path
    smooth_waypoints = MsgWaypoints()
    for i in smooth:
        smooth_waypoints.add(ned=waypoints.ned[i], airspeed=waypoints.airspeed[i],
                             course=waypoints.course[i])

    # append extra to the end to make line follower happy
    smooth_waypoints.add(ned=waypoints.ned[-1] * 1.5, airspeed=waypoints.airspeed[-1],
                         course=waypoints.course[-1])

    smooth_waypoints.add(ned=waypoints.ned[-1] * 1.5, airspeed=waypoints.airspeed[-1],
                         course=waypoints.course[-1])

    return smooth_waypoints


def find_minimum_path(tree, end_pose):
    # find the lowest cost path to the end node
    # find nodes that connect to end_node
    connecting_nodes = [tree.num_waypoints - 1]

    # # find minimum cost last node
    # idx =
    #
    # # construct lowest cost path order
    # path =

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


def random_pose(world_map=MsgWorldMap(), pd=100):
    # generate a random pose
    pose = np.random.random_sample((3,))
    pose[0:2] = pose[0:2] * world_map.city_width
    pose[2] = -pd  # height_above_ground(world_map, pose[0:2])
    return pose


def distance(start_pose, end_pose):
    # compute distance between start and end pose
    d = norm(start_pose - end_pose)
    return d


def collision2(start_pose, end_pose, world_map=MsgWorldMap()):
    # check to see of path from start_pose to end_pose colliding with map
    mid_pose = (start_pose + end_pose) / 2
    hmin = min(-start_pose[2], -end_pose[2])  # get minimum altitude over path
    w = world_map.building_width / 2  # half of the building width

    hgap = 20
    dgap = 0 + w
    # figure out which buildings to check for collisions: use the closest buildings to the
    # start, end, and mid points.

    building_index = {find_closest_building(start_pose, world_map), find_closest_building(mid_pose, world_map),
                      find_closest_building(end_pose, world_map)}
    building_index = [i for i in building_index]
    collision_flag = False

    for i in building_index:
        detail = world_map.building_details[i]
        xb = detail[0]
        yb = detail[1]
        hb = detail[2]

        if hmin - hb < hgap:  # if higher then this we can skip
            xe = end_pose[0] - xb
            ye = end_pose[1] - yb
            xs = start_pose[0] - xb
            ys = start_pose[1] - yb

            # check if any points are interior to a building
            if np.min(np.abs(np.array([xe, ye, xs, ys]))) < dgap:
                collision_flag += True
            # check if both points are on one side of an edge
            elif (xe > dgap and xs > dgap) or (xe < -dgap and xs < -dgap) or (ye > dgap and ys > dgap) or (ye < -dgap and ys < -dgap):
                collision_flag += False
            else:
                # find which corner is closest
                q = np.array([xe - xb, ye - yb])  # vector from middle of building to end point
                p = np.array([xe - xs, ye - ys])  # vector from start to end pose
                v = q - q @ p * p / norm(p)**2  # vector from center of building to nearest point on p
                # note here we have assumed, because of previous if statements, that v lies on p and not past the ends
                if np.min(np.abs(v)) > dgap:
                    collision_flag += False
                else:
                    collision_flag += True

        else:
            collision_flag += False

    if collision_flag:
        print("Collision Detected")
    return collision_flag


def collision(start_pose, end_pose, world_map=MsgWorldMap()):
    ps = start_pose[0:2]
    pe = end_pose[0:2]
    q = pe - ps
    qhat = q / norm(q)
    hm = min(start_pose[2], end_pose[2])

    width = world_map.building_width / 2
    max_dist = norm(np.array([width, width]))
    height_margin = 30
    width_margin = 25

    for b in world_map.building_details:
        if hm - b[2] < height_margin:  # check if we are higher than building
            pb = b[0:2]
            qe = pe - pb  # vector from center to end point
            qs = ps - pb  # vector from center to start
            if min(norm(qe), norm(qs)) < width + width_margin:
                print("collision detected 1: ", np.min(np.abs(np.hstack([qe, qs]))))
                return True
            v = qe - qe @ qhat * qhat
            if norm(v) > max_dist:
                pass
            else:
                # Check if v is on the line between ps and pe
                pv = v + pb
                if (ps - pv) @ qhat < 0 and (pe - pv) @ qhat > 0:
                    if np.min(np.abs(v)) < width + width_margin:
                        print("collision detected 2: ", np.min(np.abs(v)))
                        return True

    return False

def find_closest_building(p=np.array([0, 0, 0]), world_map=MsgWorldMap()):
    # find the nearest building to a point and return the index
    index = [0, 0]
    dist = np.inf
    for i, detail in enumerate(world_map.building_details):
        D = norm(p[0:2] - detail[0:2])
        if D < dist:
            dist = D
            index[1] = index[0]
            index[0] = i
    return index


def height_above_ground(world_map, point):
    # find the altitude of point above ground level
    point_height = 0
    map_height = 0
    h_agl = 0
    return h_agl


def draw_tree(tree, color, window):
    R = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, -1]])
    points = R @ tree.ned
    for i in range(points.shape[1]):
        line_color = np.tile(color, (2, 1))
        parent = int(tree.parent.item(i))
        line_pts = np.concatenate((column(points, i).T, column(points, parent).T), axis=0)
        line = gl.GLLinePlotItem(pos=line_pts,
                                 color=line_color,
                                 width=2,
                                 antialias=True,
                                 mode='line_strip')
        window.addItem(line)


def points_along_path(start_pose, end_pose, N):
    # returns points along path separated by Del
    x = np.linspace(start_pose[0], end_pose[0], N)
    y = np.linspace(start_pose[1], end_pose[1], N)
    z = np.linspace(start_pose[2], end_pose[2], N)
    points = np.hstack((x[None].T, y[None].T, z[None].T))
    return points


# WHAT IS THIS??? NUMPY INDEXING OUT OF FASHION?
def column(A, i):
    # extracts the ith column of A and return column vector
    tmp = A[:, i]
    col = tmp.reshape(A.shape[0], 1)
    return col


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
