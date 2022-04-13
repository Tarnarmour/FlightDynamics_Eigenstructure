"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
part of mavPySim 
    - Beard & McLain, PUP, 2012
    - Update history:  
        12/20/2018 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from numpy import sin, cos, tan, sqrt

# load message types
from message_types.msg_state import MsgState

import parameters.aerosonde_parameters as MAV
from tools.rotations import Quaternion2Rotation, Quaternion2Euler

# import chap4.aerosonde_parameters_torque_force_test as MAV


class MavDynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.north0],  # (0)
                               [MAV.east0],   # (1)
                               [MAV.down0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0]])   # (12)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        self._update_velocity_data()
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # initialize true_state message
        self.true_state = MsgState()

    ###################################
    # public functions
    def update(self, delta, wind):
        """
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        """
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(self._state + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(self._state + time_step*k3, forces_moments)
        self._state += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)

        # update the message class for the true state
        self._update_true_state()

    def external_set_state(self, new_state):
        self._state = new_state

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        # north = state.item(0)
        # east = state.item(1)
        # down = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        M_pos = np.array([[e1 ** 2 + e0 ** 2 - e2 ** 2 - e3 ** 2, 2 * (e1 * e2 - e3 * e0), 2 * (e1 * e3 + e2 * e0)],
                          [2 * (e1 * e2 + e3 * e0), e2 ** 2 + e0 ** 2 - e1 ** 2 - e3 ** 2, 2 * (e2 * e3 - e1 * e0)],
                          [2 * (e1 * e3 - e2 * e0), 2 * (e2 * e3 + e1 * e0), e3 ** 2 + e0 ** 2 - e1 ** 2 - e2 ** 2]])

        M_rot = np.array([[0, -p, -1, -r],
                          [p, 0, r, -q],
                          [q, -r, 0, p],
                          [r, q, -p, 0]])

        # position kinematics
        pos_dot = M_pos @ np.array([[u], [v], [w]])
        north_dot = pos_dot[0][0]
        east_dot = pos_dot[1][0]
        down_dot = pos_dot[2][0]

        # position dynamics
        u_dot = r * v - q * w + fx / MAV.mass
        v_dot = p * w - r * u + fy / MAV.mass
        w_dot = q * u - p * v + fz / MAV.mass

        # rotational kinematics
        e_dot = 0.5 * M_rot @ np.array([[e0], [e1], [e2], [e3]])
        e0_dot = e_dot[0][0]
        e1_dot = e_dot[1][0]
        e2_dot = e_dot[2][0]
        e3_dot = e_dot[3][0]

        # rotatonal dynamics
        p_dot = MAV.gamma1 * p * q - MAV.gamma2 * q * r + MAV.gamma3 * l + MAV.gamma4 * n
        q_dot = MAV.gamma5 * p * r - MAV.gamma6 * (p ** 2 - r ** 2) + m / MAV.Jy
        r_dot = MAV.gamma7 * p * q - MAV.gamma1 * q * r + MAV.gamma4 * l + MAV.gamma8 * n

        # collect the derivative of the states
        x_dot = np.array([[north_dot, east_dot, down_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3, 0]
        gust = wind[3:6, 0]
        # convert wind vector from world to body frame and add gust
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        wind_body_frame = Rb_v(phi, theta, psi) @ (steady_state + gust)
        # velocity vector relative to the airmass
        v_air = self._state[3:6, 0] - wind_body_frame
        ur = v_air[0]
        vr = v_air[1]
        wr = v_air[2]
        # compute airspeed
        self._Va = np.linalg.norm(v_air)
        # compute angle of attack
        if ur == 0:
            self._alpha = np.pi / 2
        else:
            self._alpha = np.arctan2(wr, ur)
        # compute sideslip angle
        if self._Va == 0:
            self._beta = 0
        else:
            self._beta = vr / self._Va

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        delta_a = delta.aileron
        delta_e = delta.elevator
        delta_r = delta.rudder
        delta_t = delta.throttle

        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)

        # compute gravitaional forces
        f_g = MAV.mass * MAV.gravity * np.array([[-sin(theta)], [cos(theta)*sin(phi)], [cos(theta)*cos(phi)]])

        # compute Lift and Drag coefficients
        sigma = (1 + np.exp(-MAV.M*(self._alpha - MAV.alpha0)) + np.exp(MAV.M*(self._alpha + MAV.alpha0))) / ((1 + np.exp(-MAV.M*(self._alpha - MAV.alpha0))) * (1 + np.exp(MAV.M*(self._alpha + MAV.alpha0))))
        flat_plate_model = 2*np.sign(self._alpha)*sin(self._alpha)**2 * cos(self._alpha)
        CL_alpha = (1 - sigma) * (MAV.C_L_0 + MAV.C_L_alpha * self._alpha) + sigma * flat_plate_model
        if abs(self._Va) > 0:
            CL_q = MAV.C_L_q * MAV.c / (2 * self._Va) * q
        else:
            CL_q = 0
        CL_delta = MAV.C_L_delta_e * delta_e
        CL = CL_alpha + CL_q + CL_delta

        CD_alpha = MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha * self._alpha)**2 / (np.pi * MAV.e * MAV.AR)
        if abs(self._Va) > 0:
            CD_q = MAV.C_D_q * MAV.c / (2 * self._Va) * q
        else:
            CD_q = 0
        CD_delta = MAV.C_D_delta_e * delta_e
        CD = CD_alpha + CD_q + CD_delta
        # compute Lift and Drag Forces
        F_lift = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * CL
        F_drag = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * CD

        #compute propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta_t)

        # compute longitudinal forces in body frame
        fx = sin(self._alpha) * F_lift - cos(self._alpha) * F_drag + thrust_prop + f_g[0, 0]
        fz = -sin(self._alpha) * F_drag - cos(self._alpha) * F_lift + f_g[2, 0]

        # compute lateral forces in body frame
        CY_beta = MAV.C_Y_0 + MAV.C_Y_beta * self._beta
        if abs(self._Va) > 0:
            CY_p = MAV.C_Y_p * MAV.b / (2*self._Va) * p
            CY_r = MAV.C_Y_r * MAV.b / (2 * self._Va) * r
        else:
            CY_p = 0
            CY_r = 0

        CY_delta = MAV.C_Y_delta_a * delta_a + MAV.C_Y_delta_r * delta_r
        CY = CY_beta + CY_p + CY_r + CY_delta
        fy = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * CY + f_g[1, 0]

        # compute logitudinal torque in body frame
        CM_alpha = MAV.C_m_0 + MAV.C_m_alpha * self._alpha
        if abs(self._Va) > 0:
            CM_q = MAV.C_m_q * MAV.c / (2 * self._Va) * q
        else:
            CM_q = 0
        CM_delta = MAV.C_m_delta_e * delta_e
        CM = CM_alpha + CM_q + CM_delta
        My = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * MAV.c * CM

        # compute lateral torques in body frame
        Cn_beta = MAV.C_n_0 + MAV.C_n_beta * self._beta
        if abs(self._Va) > 0:
            Cn_p = MAV.C_n_p * MAV.b / (2 * self._Va) * p
            Cn_r = MAV.C_n_r * MAV.b / (2 * self._Va) * r
        else:
            Cn_p = 0
            Cn_r = 0
        Cn_delta = MAV.C_n_delta_a * delta_a + MAV.C_n_delta_r * delta_r
        Cn = Cn_beta + Cn_p + Cn_r + Cn_delta
        Mz = 0.5 * MAV.rho * self._Va ** 2 * MAV.S_wing * MAV.b * Cn

        Cl_beta = MAV.C_ell_0 + MAV.C_ell_beta * self._beta
        if abs(self._Va) > 0:
            Cl_p = MAV.C_ell_p * MAV.b / (2 * self._Va) * p
            Cl_r = MAV.C_ell_r * MAV.b / (2 * self._Va) * r
        else:
            Cl_p = 0
            Cl_r = 0

        Cl_delta = MAV.C_ell_delta_a * delta_a + MAV.C_ell_delta_r * delta_r
        Cl = Cl_beta + Cl_p + Cl_r + Cl_delta
        Mx = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * MAV.b * Cl + torque_prop

        self._forces[0] = fx
        self._forces[1] = fy
        self._forces[2] = fz

        output = np.array([[fx, fy, fz, Mx, My, Mz]]).T

        return output

    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller  (See addendum by McLain)
        # map delta_t throttle command(0 to 1) into motor input voltage
        V_in = MAV.V_max * delta_t

        a = MAV.C_Q0 * MAV.rho * MAV.D_prop**5 / (4 * np.pi**2)
        b = MAV.C_Q1 * MAV.rho * MAV.D_prop**4 / (2 * np.pi) * Va + MAV.KQ**2 / MAV.R_motor
        c = MAV.C_Q2 * MAV.rho * MAV.D_prop**3 * Va**2 - (MAV.KQ / MAV.R_motor) * V_in + MAV.KQ * MAV.i0

        # Angular speed of propeller
        Omega_p = (-b + sqrt(b**2 - 4*a*c)) / (2*a)
        J_op = 2 * np.pi * Va / (Omega_p * MAV.D_prop)
        rps = Omega_p / (2 * np.pi)
        # thrust and torque due to propeller
        CT = MAV.C_T2 * J_op**2 + MAV.C_T1 * J_op + MAV.C_T0
        CQ = MAV.C_Q2 * J_op**2 + MAV.C_Q1 * J_op + MAV.C_Q0
        thrust_prop = MAV.rho * rps**2 * MAV.D_prop**4 * CT
        torque_prop = -MAV.rho * rps**2 * MAV.D_prop**5 * CQ
        return thrust_prop, torque_prop

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        pdot = Quaternion2Rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)

def Rb_v(phi, theta, psi):
    return np.array([[cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta)],
                     [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), sin(phi)*cos(theta)],
                     [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi), cos(phi)*cos(theta)]]
                    )
