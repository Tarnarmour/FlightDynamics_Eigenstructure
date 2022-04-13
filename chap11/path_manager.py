import numpy as np
from numpy import sin, cos, sqrt, tan
from numpy.linalg import norm
import sys
sys.path.append('..')
from chap11.dubins_parameters import DubinsParameters
from message_types.msg_path import MsgPath
from message_types.msg_waypoints import MsgWaypoints
from parameters.aerosonde_parameters import Va0


class PathManager:
    def __init__(self):
        # message sent to path follower
        self.path = MsgPath()
        self.path.airspeed = Va0
        # pointers to previous, current, and next waypoints
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2
        self.num_waypoints = 0
        self.halfspace_n = np.inf * np.ones((3,))
        self.halfspace_r = np.inf * np.ones((3,))
        # state of the manager state machine
        self.manager_state = 0
        self.manager_requests_waypoints = True
        self.dubins_path = DubinsParameters()

    def update(self, waypoints, radius, state):
        if waypoints.idle == True:
            path = self.get_idle_orbit(radius, state)
            return path

        if waypoints.num_waypoints == 0:
            self.manager_requests_waypoints = True
        if self.manager_requests_waypoints is True \
                and waypoints.flag_waypoints_changed is True:
            self.manager_requests_waypoints = False
            self.num_waypoints = waypoints.num_waypoints
        if waypoints.type == 'straight_line':
            self.line_manager(waypoints, state)
        elif waypoints.type == 'fillet':
            self.fillet_manager(waypoints, radius, state)
        elif waypoints.type == 'dubins':
            self.dubins_manager(waypoints, radius, state)
        else:
            print('Error in Path Manager: Undefined waypoint type.')
        self.path.orbit_radius = radius
        return self.path

    def get_idle_orbit(self, radius, state):
        path = MsgPath()
        path.type = 'orbit'
        path.orbit_center = np.array([state.north, state.east, state.altitude])
        path.orbit_radius = radius
        path.orbit_direction = 'CW'
        path.plot_updated = False
        path.airspeed = Va0
        return path

    def initialize_pointers(self):
        if self.num_waypoints >= 3:
            self.ptr_previous = self.ptr_current
            self.ptr_current = self.ptr_next
            self.ptr_next = self.ptr_next + 1
        else:
            print('Error Path Manager: need at least three waypoints')

    def increment_pointers(self):
        self.ptr_previous = self.ptr_current
        self.ptr_current = self.ptr_next
        self.ptr_next = self.ptr_next + 1
        if self.ptr_next >= self.num_waypoints - 1:
            self.manager_requests_waypoints = True

    def inHalfSpace(self, pos, waypoints=MsgWaypoints()):

        p0 = np.copy(waypoints.ned[self.ptr_previous])
        p1 = np.copy(waypoints.ned[self.ptr_current])
        p2 = np.copy(waypoints.ned[self.ptr_next])

        q1 = p1 - p0
        q2 = p2 - p1

        a = np.cross(-q1, q2)
        b = q2 - q1
        n = np.cross(a, b)

        r = pos - p1

        if (r @ n > 0): #implement code here
            return True
        else:
            return False

    def line_manager(self, waypoints, state):
        pMav = np.array([state.north, state.east, -state.altitude]) # current MAV location

        p0 = np.copy(waypoints.ned[self.ptr_previous])
        p1 = np.copy(waypoints.ned[self.ptr_current])
        p2 = np.copy(waypoints.ned[self.ptr_next])

        if self.inHalfSpace(pMav, waypoints):  # Half-way plane check
            self.increment_pointers()

            self.path.type = 'line'
            self.path.airspeed = 25
            self.path.line_origin = p1
            self.path.line_direction = (p2 - p1) / np.linalg.norm(p2 - p1)
            self.path.plot_updated = False
        else:
            self.path.type = 'line'
            self.path.airspeed = 25
            self.path.line_origin = p0
            self.path.line_direction = (p1 - p0) / np.linalg.norm(p1 - p0)

    def fillet_manager(self, waypoints, radius, state):
        p_mav = np.array([state.north, state.east, state.altitude])

        p0 = np.copy(waypoints.ned[self.ptr_previous])
        p1 = np.copy(waypoints.ned[self.ptr_current])

        if waypoints.num_waypoints - 1 < self.ptr_next:
            p2 = 2 * p1 - p0
        else:
            p2 = np.copy(waypoints.ned[self.ptr_next])

        q1 = p1 - p0
        q2 = p2 - p1

        # find q2 rotated into the q1 frame (so that q1 = [1 0 0], and we can find varphi more easily)
        th = np.arctan2(q1[1], q1[0])
        R = rot(th)
        e2 = R.T @ -q2

        # varphi is now just arctan(y, x)
        varphi = np.arctan2(e2[1], e2[0])

        # state machine: 0 = first time, 1 = following straight line, 2 = following orbit
        if self.manager_state == 0:
            # initialize stuff the first time through, assume we are on a straight line
            self.path = self.construct_fillet_line(p0, p1)
            self.manager_state = 1
        elif self.manager_state == 1:
            # we are following a straight line, check if we have reached the end
            ec = p_mav - p0  # current MAV location relative to waypoint 0
            dist_along_q1 = ec @ q1 / norm(q1)
            c = radius / abs(tan(varphi / 2))
            stopping_distance = norm(q1) - c  # c is the distance from the orbit between the lines

            if dist_along_q1 > stopping_distance:
                # if we have reached end of line segment, start orbit
                # b is the bisector of q1 and q2, found by rotating q2 halfway to -q1
                b = rot(varphi / 2).T @ (q2 / norm(q2))
                self.path = self.construct_fillet_circle(p1, b, varphi, radius)
                self.manager_state = 2
            else:
                pass
        elif self.manager_state == 2:
            # we are on the orbit path, and need to check if we've reached the end
            ec = p_mav - p1  # current MAV location relative to waypoint 1
            dist_along_q2 = ec @ q2 / norm(q2)
            c = radius / abs(tan(varphi / 2))
            stopping_distance = c

            if dist_along_q2 > stopping_distance:
                # we have reached the end of the orbit
                self.path = self.construct_fillet_line(p1, p2)
                self.increment_pointers()
                self.manager_state = 1
            else:
                pass

    def construct_fillet_line(self, p0, p1):
        path = MsgPath()
        path.airspeed = self.path.airspeed
        path.type = 'line'
        q = (p1 - p0) / np.linalg.norm(p1 - p0)
        path.line_direction = q
        path.line_origin = p0
        path.airspeed = self.path.airspeed
        path.plot_updated = False  # update the plot
        return path

    def construct_fillet_circle(self, p1, b, varphi, radius):
        path = MsgPath()
        path.airspeed = self.path.airspeed
        path.type = 'orbit'
        if varphi > 0:
            path.orbit_direction = 'CW'
        else:
            path.orbit_direction = 'CCW'
        path.orbit_center = p1 + b * radius / abs(sin(varphi / 2))
        path.orbit_radius = radius
        path.plot_updated = False
        return path

    def dubins_manager(self, waypoints, radius, state):
        p_mav = np.array([state.north, state.east, -state.altitude])
        chi_mav = state.chi

        # 0 - initialize
        # 1 - finished path, generate new dubins param
        # 2 - generate first circle -- check if we are close enough to skip to line
        # 3 - assume we are on the wrong side of the half plane, wait till passing
        # 4 - wait till passing half plane on the right side, generate line path
        # 5 - follow line, check for half plane. Generate end circle, if close enough to end just finish
        # 6 - assume wrong side of half plane for end circle, wait for check
        # 7 - wait for other side of half plane, reset to 1

        # if the waypoints have changed, update the waypoint pointer
        if self.manager_state == 0:
            print("Current State: 0")
            # first pass, initialize
            self.manager_state = 1
            print("Current State: 1")
        if self.manager_state == 1:
            # just finished dubins path, generate a new dubins path
            p0 = waypoints.ned[self.ptr_previous]
            p1 = waypoints.ned[self.ptr_current]
            p2 = waypoints.ned[self.ptr_next]

            q1 = p1 - p0
            q2 = p2 - p1

            chis = np.arctan2(q1[1], q1[0])
            chie = np.arctan2(q2[1], q2[0])

            self.dubins_path = DubinsParameters(p0, chis, p1, chie, radius)
            self.manager_state = 2
            print("Current State: 2")

        if self.manager_state == 2:
            # take dubins path and generate first circle
            ec = p_mav - self.dubins_path.r1
            if norm(ec) < 50:
                self.path = self.construct_dubins_line(waypoints, self.dubins_path)
                self.manager_state = 5
                print("Current State: 5")
            else:
                self.path = self.construct_dubins_circle_start(waypoints, self.dubins_path)
                self.manager_state = 3
                print("Current State: 3")

        if self.manager_state == 3:
            # assume we are on the wrong side of the first half plane and wait to pass
            ec = p_mav - self.dubins_path.r1
            if ec @ self.dubins_path.n1 < 0:
                self.manager_state = 4
                print("Current State: 4")

        if self.manager_state == 4:
            # check if we have finished the first dubins circle, as in dubins_path
            # if we have finished, step state to 4 and set path to line segment
            ec = p_mav - self.dubins_path.r1  # MAV pos relative to first half plane
            if ec @ self.dubins_path.n1 > 0:
                self.path = self.construct_dubins_line(waypoints, self.dubins_path)
                self.manager_state = 5
                print("Current State: 5")

        if self.manager_state == 5:
            # check if we have finished the straight line segment
            # if we have finished, start the dubins end circle
            ec = p_mav - self.dubins_path.r2
            if ec @ self.dubins_path.n1 > 0:
                ec = p_mav - self.dubins_path.r3
                if norm(ec) < 50:  # close enough to just finish
                    self.manager_state = 1
                    print("Current State: 1")
                    self.increment_pointers()
                    self.dubins_manager(waypoints, radius, state)
                else:
                    self.path = self.construct_dubins_circle_end(waypoints, self.dubins_path)
                    self.manager_state = 6
                    print("Current State: 6")

        if self.manager_state == 6:
            # ASSUME we are on wrong side of final dubins circle
            ec = p_mav - self.dubins_path.r3
            if ec @ self.dubins_path.n3 < 0:
                self.manager_state = 7
                print("Current State: 7")

        if self.manager_state == 7:
            # check if we have finished the end dubins circle, as in dubins_path
            # if we have finished, step state to 1 and update waypoints
            ec = p_mav - self.dubins_path.r3
            if ec @ self.dubins_path.n3 > 0:  # and mod(abs(self.dubins_path.chi_e - chi_mav)) < np.pi / 4:
                self.increment_pointers()
                self.manager_state = 1
                print("Current State: 1")
                self.dubins_manager(waypoints, radius, state)

    def construct_dubins_circle_start(self, waypoints, dubins_path):
        path = MsgPath()
        path.airspeed = self.path.airspeed
        path.type = 'orbit'
        path.orbit_center = dubins_path.center_s
        path.orbit_radius = dubins_path.radius
        if dubins_path.dir_s == 1:
            dir = 'CCW'
        else:
            dir = 'CW'
        path.orbit_direction = dir
        path.plot_updated = False
        return path

    def construct_dubins_line(self, waypoints, dubins_path):
        path = MsgPath()
        path.airspeed = self.path.airspeed
        path.type = 'line'
        path.line_origin = dubins_path.r1
        path.line_direction = (dubins_path.r2 - dubins_path.r1) / norm(dubins_path.r2 - dubins_path.r1)
        path.plot_updated = False
        return path

    def construct_dubins_circle_end(self, waypoints, dubins_path):
        path = MsgPath()
        path.airspeed = self.path.airspeed
        path.type = 'orbit'
        path.orbit_center = dubins_path.center_e
        path.orbit_radius = dubins_path.radius
        if dubins_path.dir_e == 1:
            dir = 'CCW'
        else:
            dir = 'CW'
        path.orbit_direction = dir
        path.plot_updated = False
        return path

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