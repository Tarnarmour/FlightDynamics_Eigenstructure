import numpy as np
from numpy import sin, cos, tan, sqrt, pi
from numpy.linalg import norm
import matplotlib.pyplot as plt

R = np.array([1, 0, 0, 1])
B = np.array([0, 0, 1, 1])
G = np.array([0, 1, 0, 1])
K = np.array([0, 0, 0, 1])
Y = np.array([0.5, 0.5, 0, 1])

from message_types.msg_world_map import MsgWorldMap
from chap11.dubins_parameters import DubinsParameters
from parameters.planner_parameters import R_min, Va0
R_min = 20
from message_types.msg_waypoints import MsgWaypoints
from tools.wrap import wrap

world_map = MsgWorldMap()
fig, ax = plt.subplots()
segment_length = 200.0

def draw_building(pos=np.array([0, 0]), w=20, color=np.array([0.2, 0.2, 0.2, 1])):
    xs = np.array([-w / 2, w / 2, w / 2, -w / 2, -w / 2]) + pos[0]
    ys = np.array([-w / 2, -w / 2, w / 2, w / 2, -w / 2]) + pos[1]
    ax.plot(xs, ys, color=color)


def scatter(pose, color=B):
    x, y = pose[0], pose[1]
    handle = ax.scatter(x, y, color=color)
    return handle


def draw_dubins(points, color=B):
    handle, = ax.plot(points[:, 0], points[:, 1], color=color)
    return handle


def check_collision(points, world_map=MsgWorldMap()):
    height_margin = 0
    width_margin = 0
    width = world_map.building_width
    for p in points:
        for b in world_map.building_details:
            if p[2] - b[2] < height_margin:
                v = p[0:2] - b[0:2]
                if np.max(np.abs(v)) < width_margin + width:
                    return True
    return False


def random_pose(world_map=MsgWorldMap(), h=0.0):
    p = np.random.random_sample((3,))
    p[2] = h
    w = world_map.city_width
    p[0:2] = p[0:2] * w * 1.2 - 0.1 * w
    chi = np.random.random_sample((1,))[0] * 2 * pi - pi
    return p, chi


def find_nearest_node(p_star, chi_star, tree=MsgWaypoints()):
    # dubins = DubinsParameters()
    # Lmax = np.inf
    # index = 0
    # for i in range(tree.num_waypoints):
    #     p_prev, chi_prev = tree.ned[i], tree.course[i]
    #     print(p_prev, chi_prev, p_star, chi_star, R_min)
    #     dubins.update(p_prev, chi_prev, p_star, chi_star, R_min)
    #     if dubins.length < Lmax:
    #         Lmax = dubins.length
    #         index = i
    # return index
    dist = np.inf
    index = 0
    for i, p in enumerate(tree.ned):
        D = abs(norm(p - p_star) - 3*R_min)
        if D < dist:
            index = i
            dist = D
    return index



def find_valid_node(p1, p2, radius=R_min):
    v = p2 - p1
    d = norm(v)
    q = v / d
    L = max(segment_length, 4.1 * radius)
    p_star = p1 + q * L
    # if d < 3 * radius:
    #     p_star = p1 + q * 3.1 * radius
    # else:
    #     p_star = p2
    return p_star


def check_finished(tree, end_pose, max_length):
    p_last, p_end = tree.ned[-1], end_pose
    if norm(p_last - p_end) < max_length:
        return True
    else:
        return False


def rot(th):
    return np.array([[cos(th), -sin(th), 0],
                     [sin(th), cos(th), 0],
                     [0, 0, 1]])


def mod(x):
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x


def get_dubins_points(dubins_path=DubinsParameters(), dx=1.0):
    ps = dubins_path.p_s
    pe = dubins_path.p_e
    cs = dubins_path.center_s
    ce = dubins_path.center_e
    r1 = dubins_path.r1
    r2 = dubins_path.r2
    r3 = dubins_path.r3
    radius = dubins_path.radius
    lams = dubins_path.dir_s
    lame = dubins_path.dir_e

    # loop through first circle
    q1 = ps - cs
    q2 = r1 - cs
    phi1 = np.arctan2(q1[1], q1[0])
    phi2 = np.arctan2(q2[1], q2[0])
    if lams == -1:
        if phi2 > phi1:
            phis = phi2 - phi1 - 2*pi
        else:
            phis = phi2 - phi1
    elif lams == 1:
        if phi2 < phi1:
            phis = phi2 - phi1 + 2 * pi
        else:
            phis = phi2 - phi1
    else:
        phis = None
        print("error in lams")
    arc_length_s = abs(phis) * radius
    if np.isnan(arc_length_s):
        print("error here")
    N_s = int(arc_length_s / dx)

    # loop through first straight line
    v = r2 - r1
    length_straight = norm(v)
    N_line = int(length_straight / dx)

    # loop through second circle
    q3 = r2 - ce
    q4 = r3 - ce
    phi3 = np.arctan2(q3[1], q3[0])
    phi4 = np.arctan2(q4[1], q4[0])
    if lame == -1:
        if phi4 > phi3:
            phie = phi4 - phi3 - 2 * pi
        else:
            phie = phi4 - phi3
    elif lame == 1:
        if phi4 < phi3:
            phie = phi4 - phi3 + 2 * pi
        else:
            phie = phi4 - phi3
    else:
        phie = None
        print("error in lame")
    arc_length_e = abs(phie) * radius
    N_e = int(arc_length_e / dx)

    N = N_s + N_line + N_e
    points = np.zeros((N, 3))

    for i in range(N_s):
        theta = phi1 + phis * i / N_s
        points[i] = cs + radius * rot(theta) @ np.array([1, 0, 0])

    for i in range(N_line):
        points[N_s + i] = r1 + v * i / N_line

    for i in range(N_e):
        theta = phi3 + phie * i / N_e
        points[N_line + N_s + i] = ce + radius * rot(theta) @ np.array([1, 0, 0])

    print("phi_s: ", phis, " phi_e: ", phie)
    return points


for b in world_map.building_details:
    draw_building(b[0:2], world_map.building_width, np.array([0.2, 0.2, 0.2, 1]))

# plt.xlim(-800, 400)
# plt.ylim(-600, 600)
# plt.axis('equal')

start_pose = np.array([0, 0, 0])
chi_s = pi/4
end_pose = np.array([2000, 2000, 0])
chi_e = pi/2

scatter(start_pose, color=K)
scatter(end_pose, color=K)

tree = MsgWaypoints()
tree.type = 'dubins'

# dubins_path = DubinsParameters(start_pose, chi_s, end_pose, chi_e, R_min)
# ax.scatter(dubins_path.r1[0], dubins_path.r1[1], color=Y)
# ax.scatter(dubins_path.r2[0], dubins_path.r2[1], color=Y)
# ax.scatter(dubins_path.center_s[0], dubins_path.center_s[1], color=R)
# ax.scatter(dubins_path.center_e[0], dubins_path.center_e[1], color=R)
# points = get_dubins_points(dubins_path, dx=1.0)
# flag = check_collision(points, world_map)
# print("dir_s: ", dubins_path.dir_s, "dir_e: ", dubins_path.dir_e)
# print("Collision: ", flag)
# draw_dubins(points)

tree.add(ned=start_pose, course=chi_s, parent=0, cost=0.0, airspeed=Va0)

while not check_finished(tree, end_pose, segment_length):
    if np.random.randint(0, 100) > 10:
        p_star, chi_star = random_pose(world_map, h=0.0)
    else:
        p_star, chi_star = end_pose, chi_e
    index = find_nearest_node(p_star, chi_star, tree)
    p_last, chi_last = tree.ned[index], tree.course[index]
    p_star = find_valid_node(p_last, p_star, R_min)
    dubins_path = DubinsParameters(p_last, chi_last, p_star, chi_star, R_min)
    points = get_dubins_points(dubins_path, dx=1.0)
    handle = draw_dubins(points, Y)
    plt.pause(0.001)
    collision = check_collision(points, world_map)
    if collision:
        handle.remove()
        print("collision detected")
    else:
        handle.remove()
        draw_dubins(points, B)
        scatter(p_star, K)
        cost = tree.cost[index] + 1
        tree.add(ned=p_star, course=chi_star, parent=index, cost=cost, airspeed=Va0)
    plt.pause(0.1)

p_last = tree.ned[-1]
chi_last = tree.course[-1]
cost_prev = tree.cost[-1]
parent = tree.parent[-1]

q = (end_pose - p_last) / norm(end_pose - p_last)
p_star = p_last + q * max(norm(end_pose - p_last), 4.1 * R_min)
chi_star = chi_e

dubins_path = DubinsParameters(p_last, chi_last, p_star, chi_star, R_min)
points = get_dubins_points(dubins_path, dx=1.0)
draw_dubins(points, B)
plt.pause(0.1)
tree.add(ned=p_star, course=chi_star, parent=parent, cost=cost_prev+1, airspeed=Va0)

plt.show()






