import numpy as np
import parameters.control_parameters as AP
from tools.transfer_function import transferFunction
from tools.wrap import wrap
from chap6.pi_control import PIControl
from chap6.pd_control_with_rate import PDControlWithRate
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from message_types.msg_autopilot import MsgAutopilot
from tools.rotations import Quaternion2Rotation, Rotation2Quaternion, Euler2Quaternion, Quaternion2Euler
from chap5.model_coef import A, A_lat, A_lon, B, B_lat, B_lon, u_trim, x_trim
import control as ct
from project.eigenstructure_assignment import assign


class AutopilotEig:
    def __init__(self, ts_control):
        self.commanded_state = MsgState()
        self.ts = ts_control
        self.x_eq = self.state_to_x(x_trim)
        self.u_eq = np.squeeze(u_trim)

        """States: pn = north, pe = east, pd = down, u = vel x, v = vel y, w = vel z, phi = roll, theta = pitch, psi = yaw, p = roll rate, q = pitch rate, r = yaw rate"""
        """            0            1        2          3            4         5           6            7           8           9              10               11"""
        self.A = A[2:None, 2:None]
        self.B = B[2:None, :]
        self.Cr = np.array([[1]])

        open_poles = [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, -22.4411 + 0.0000j, -4.8778 + 9.8690j,
                      -4.8778 - 9.8690j, -1.1402 + 4.6549j, -1.1402 - 4.6549j, -0.1175 + 0.4925j,
                      -0.1175 - 0.4925j, 0.1020 + 0.0000j, 0.0113 + 0.0000j]
        des_poles = [-3, -25, - 8 + 5j, - 8 - 5j, - 5 + 2j, - 5 - 2j, - 7 + 0.2j, - 7 - 0.2j, - 5 + 0.1j, - 5 - 0.1j]
        self.K = np.squeeze(np.asarray(ct.place(self.A, self.B, des_poles)))

        self.A_long = A_lon
        self.B_long = B_lon
        self.Cr_long = np.array([[1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, -1]])

        self.A_lat = A_lat
        self.B_lat = B_lat
        self.Cr_lat = np.array([[1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1]])

        """Eigenstructure Assignment: Longitudinal"""
        """Longitudinal State: [u, w, q, theta, pd] u = north velocity, w = up velocity, q = body frame y angular velocity, theta = pitch angle, pd = -altitude]"""
        open_long_poles = [0.0000 + 0.0000j, -4.8778 + 9.8690j, -4.8778 - 9.8690j, -0.1175 + 0.4925j, -0.1175 - 0.4925j]
        des_long_poles = [-1.0, -10 + 3j, -10 - 3j, - 2 + 2j, - 2 - 2j]

        V_long_des = np.array([[1, 0, 0, 0, 0],
                               [0, 0.925, -0.015 + 0.3j, 0.03 + 0.015j, -0.02 + 0.025j],
                               [0, 0.925, -0.015 - 0.3j, 0.03 - 0.015j, -0.02 - 0.025j],
                               [0, 1+1j, 0, 0, 0],
                               [0, 1-1j, 0, 0, 0]]).T

        D_long = [np.array([[0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0]]),
                  np.array([[1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0]]),
                  np.array([[1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0]]),
                  np.array([[1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0]]),
                  np.array([[1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0]])
                  ]

        # Pole Placement
        self.K_long = np.squeeze(np.asarray(ct.place(self.A_long, self.B_long, des_long_poles)))
        self.kr_long = -1 * np.linalg.inv \
            (self.Cr_long @ np.linalg.inv(self.A_long - self.B_long @ self.K_long) @ self.B_long)

        # Eigenstructure Assignment
        self.K_long_ea, E_long_ea, V_long_ea = assign(self.A_long, self.B_long, des_long_poles, V_long_des, D_long)
        self.kr_long_ea = -1 * np.linalg.inv \
            (self.Cr_long @ np.linalg.inv(self.A_long - self.B_long @ self.K_long_ea) @ self.B_long)

        """Lateral State:[v, p, r, phi, psi] v = east velocity, p = roll rate, r = yaw rate, phi = roll, psi = yaw"""
        open_lat_poles = [-22.4411 + 0.0000j, -1.1402 + 4.6549j, -1.1402 - 4.6549j, 0.1021 + 0.0000j, 0.0113 + 0.0000j]
        des_lat_poles = [-30, - 5 + 5j, - 5 - 5j, -5, -0.5]
        V_lat_des = np.array([[0, 1, 0, 0, 0],  # roll subsidence
                              [1, 0.2+0.05j, 0, 0, 0.0],  # dutch roll
                              [1, 0.2-0.05j, 0, 0, 0.0],
                              [0, 0, 0, 0.3, 1.0],  # spiral?
                              [0, 0, 0, 0, 1]]).T  # spiral?

        D_lat = [np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0]]),
                 np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1]]),
                 np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1]]),
                 np.array([[0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]]),
                 np.array([[0, 0, 0, 0, 1]])
                 ]

        # Pole Placement
        self.K_lat = np.squeeze(np.asarray(ct.place(self.A_lat, self.B_lat, des_lat_poles)))
        self.kr_lat = -1 * np.linalg.inv(self.Cr_lat @ np.linalg.inv(self.A_lat - self.B_lat @ self.K_lat) @ self.B_lat)

        # Eigenstructure Assignment
        self.K_lat_ea, E_lat_ea, V_lat_ea = assign(self.A_lat, self.B_lat, des_lat_poles, V_lat_des, D_lat)
        self.kr_lat_ea = -1 * np.linalg.inv \
            (self.Cr_lat @ np.linalg.inv(self.A_lat - self.B_lat @ self.K_lat_ea) @ self.B_lat)

        self.long_ix = [3, 5, 10, 7, 2]
        self.lat_ix = [4, 9, 11, 6, 8]

        self.e_max = np.radians(45)
        self.a_max = np.radians(45)
        self.r_max = np.radians(45)
        self.t_max = 1.0
        self.t_min = 0.0

        with np.printoptions(precision=4, linewidth=500, floatmode='fixed', suppress=True, sign=' '):
            print("Longitudinal Eigenstructure Assignment:")
            print("Eigenvalues: ", E_long_ea)
            print("Desired Eigenvectors:")
            for v in V_long_des.T:
                print(v)
            print("Achieved Eigenvectors:")
            for v in V_long_ea.T:
                print(v)
            print("K:")
            print(self.K_long_ea)

            print("\nLateral Eigenstructure Assignment:")
            print("Eigenvalues: ", E_lat_ea)
            print("Desired Eigenvectors:")
            for v in V_lat_des.T:
                print(v)
            print("Achieved Eigenvectors:")
            for v in V_lat_ea.T:
                print(v)
            print("K:")
            print(self.K_lat_ea)

    def state_to_x(self, state=np.zeros((13,))):
        if len(state.shape) > 1:
            state = np.squeeze(state)
        x = np.zeros((12,))
        x[0:6] = state[0:6]
        x[6:9] = Quaternion2Euler(state[6:10])
        x[9:None] = state[10:None]
        return x

    def update(self, cmd=MsgAutopilot(), state=np.zeros((13, 1))):

        x = self.state_to_x(state) - self.x_eq
        r_psi = cmd.course_command - self.x_eq[8]
        r_u = cmd.airspeed_command - self.x_eq[3]
        r_altitude = -cmd.altitude_command - self.x_eq[2]

        r_long = np.array([r_u, r_altitude])
        r_lat = np.array([0, r_psi])

        '''[u, w, q, theta, pd]'''
        x_long = x[self.long_ix]
        x_long[-1] *= -1
        '''[v, p, r, phi, psi]'''
        x_lat = x[self.lat_ix]

        u_long = -self.K_long @ x_long + self.kr_long @ r_long
        u_lat = -self.K_lat @ x_lat + self.kr_lat @ r_lat

        u_long_ea = -self.K_long_ea @ x_long + self.kr_long_ea @ r_long
        u_lat_ea = -self.K_lat_ea @ x_lat + self.kr_lat_ea @ r_lat

        # Regular Pole Placement
        # u = np.array([u_long[0], u_lat[0], u_lat[1], u_long[1]]) + self.u_eq
        # Eigenstructure Assignment
        u = np.array([u_long_ea[0], u_lat_ea[0], u_lat_ea[1], u_long_ea[1]]) + self.u_eq

        delta_e = self.saturate(u[0], -self.e_max, self.e_max)
        delta_a = self.saturate(u[1], -self.a_max, self.a_max)
        delta_r = self.saturate(u[2], -self.r_max, self.r_max)
        delta_t = self.saturate(u[3], self.t_min, self.t_max)

        delta = MsgDelta(elevator=delta_e,
                         aileron=delta_a,
                         rudder=delta_r,
                         throttle=delta_t)

        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.phi = self.x_eq[6]
        self.commanded_state.theta = self.x_eq[7]
        self.commanded_state.chi = cmd.course_command

        return delta, self.commanded_state

    @staticmethod
    def saturate(input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
