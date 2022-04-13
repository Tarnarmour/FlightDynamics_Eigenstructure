import numpy as np
from math import sin, cos
import sys

sys.path.append('..')
from parameters.aerosonde_parameters import gravity as g
from parameters.simulation_parameters import ts_control
from message_types.msg_autopilot import MsgAutopilot
from message_types.msg_path import MsgPath
from message_types.msg_state import MsgState
from tools.wrap import wrap


class PathFollower:
    def __init__(self):
        self.chi_inf = 90 * np.pi / 180  # approach angle for large distance from straight-line path
        self.k_path = 0.05 #0.05  # proportional gain for straight-line path following
        self.k_orbit = 5 # 10.0  # proportional gain for orbit following
        self.gravity = 9.81
        self.autopilot_commands = MsgAutopilot()  # message sent to autopilot

        self.flag1 = True

        self.e_ = np.array([0., 0.])
        self.ed_ = np.array([0., 0.])
        self.beta = 0.05
        self.eint = 0.

        self.kd = 0.05
        self.ki = 0.001

    def update(self, path, state):
        if path.type == 'line':
            self._follow_straight_line(path, state)
        elif path.type == 'orbit':
            self._follow_orbit(path, state)
            self.flag1 = True
        return self.autopilot_commands

    def _follow_straight_line(self, path=MsgPath(), state=MsgState()):
        self.autopilot_commands.airspeed_command = path.airspeed
        # course command
        q = np.copy(path.line_direction)
        q[2] *= -1

        r = np.copy(path.line_origin)
        r[2] *= -1

        p = np.array([state.north, state.east, state.altitude])
        e = p - r

        chi_q = np.arctan2(q[1], q[0])
        # chi_q = wrap(chi_q, state.chi)

        Rp = np.array([[cos(chi_q), sin(chi_q), 0],
                       [-sin(chi_q), cos(chi_q), 0],
                       [0, 0, 1]])
        e_p = Rp @ e

        if self.flag1:
            self.e_ = np.array([e_p[1], e_p[1]])
            self.flag1 = False

        self.e_[0], self.e_[1] = e_p[1], self.e_[0]
        self.ed_[0], self.ed_[1] = (1 - self.beta) * self.ed_[1] + self.beta * (self.e_[0] - self.e_[1]) / ts_control, self.ed_[0]
        if abs(self.ed_[0]) < 0.1:
            self.eint = self.eint + ts_control / 2 * (np.sum(self.e_))

        chi_d = self.chi_inf * 2 / np.pi * np.arctan(self.k_path * e_p[1] + self.kd * self.ed_[0] + self.ki * self.eint)

        chi_c = chi_q - chi_d

        chi_mav = wrap(state.chi, chi_q)

        # THIS IS CURSED BLACK MAGIC, DON'T MEDDLE OR YOU'LL STAY UP TILL 2:42 AM POUNDING YOUR FIST ON THE TABLE
        if abs(wrap(chi_mav - chi_q, 0)) > np.pi / 2:
            chi_c = chi_q

        self.autopilot_commands.course_command = chi_c
        # altitude command
        p_q = e @ q * q
        self.autopilot_commands.altitude_command = r[2] + p_q[2]
        # feedforward roll angle for straight line is zero
        self.autopilot_commands.phi_feedforward = 0.

    def _follow_orbit(self, path=MsgPath(), state=MsgState()):
        pass
        if path.orbit_direction == 'CW':
            direction = 1.0
            # print('CCW')  # pay no attention to this obvious apparent mistake, it's because we
            # are using NED coordinates and CW from below looks like CCW from above
            # print(1.0)
        else:
            direction = -1.0
            # print(-1.0)
            # print('CW')
        # airspeed command
        self.autopilot_commands.airspeed_command = path.airspeed
        # distance from orbit center
        c = path.orbit_center
        rho = path.orbit_radius
        q = np.array([state.north, state.east, state.altitude])
        chi_c = state.chi
        e = c - q
        d = np.linalg.norm(e[0:2])

        # compute wrapped version of angular position on orbit
        varphi = np.arctan2(e[1], e[0])
        while varphi - chi_c < -np.pi:
            varphi += 2 * np.pi

        # compute normalized orbit error
        orbit_error = (d - rho) / rho

        # course command
        chi_0 = varphi + direction * np.pi / 2
        chi_d = -direction * np.arctan(self.k_orbit * orbit_error)
        self.autopilot_commands.course_command = chi_0 + chi_d

        # altitude command
        self.autopilot_commands.altitude_command = -c[2]

        # roll feedforward command
        if orbit_error < 10:
            self.autopilot_commands.phi_feedforward = direction * np.arctan2(state.Va**2, g * rho)
        else:
            a1 = state.wn * cos(chi_c) + state.we * sin(chi_c)
            a2 = np.sqrt(state.Va**2 - (state.wn * sin(chi_c) - state.we * cos(chi_c))**2)
            a3 = g * rho * np.sqrt((state.Va**2 - (state.wn * sin(chi_c) - state.we * cos(chi_c))**2) / (state.Va**2))
            self.autopilot_commands.phi_feedforward = direction * np.arctan2((a1 + a2)**2, a3)



