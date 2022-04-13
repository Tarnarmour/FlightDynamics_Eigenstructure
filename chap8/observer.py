"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import sys
import numpy as np
from scipy import stats
sys.path.append('..')
import parameters.control_parameters as CTRL
import parameters.aerosonde_parameters as MAV
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors
from numpy import sin, cos, tan, sqrt

class Observer:
    def __init__(self, ts_control, initial_state=MsgState(), initial_measurements = MsgSensors()):
        # initialized estimated state message
        self.estimated_state = initial_state
        self.true_state = MsgState()
        # use alpha filters to low pass filter gyros and accels
        # alpha = Ts/(Ts + tau) where tau is the LPF time constant
        self.lpf_gyro_x = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_z)
        self.lpf_accel_x = AlphaFilter(alpha=0.7, y0=initial_measurements.accel_x)
        self.lpf_accel_y = AlphaFilter(alpha=0.7, y0=initial_measurements.accel_y)
        self.lpf_accel_z = AlphaFilter(alpha=0.7, y0=initial_measurements.accel_z)
        # use alpha filters to low pass filter absolute and differential pressure
        self.lpf_abs = AlphaFilter(alpha=0.97, y0=initial_measurements.abs_pressure)
        self.lpf_diff = AlphaFilter(alpha=0.95, y0=initial_measurements.diff_pressure)
        # ekf for phi and theta
        self.attitude_ekf = EkfAttitude()
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = EkfPosition()


    def update(self, S=MsgSensors()):

        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(S.gyro_x - SENSOR.gyro_x_bias)
        self.estimated_state.q = self.lpf_gyro_y.update(S.gyro_y - SENSOR.gyro_y_bias)
        self.estimated_state.r = self.lpf_gyro_z.update(S.gyro_z - SENSOR.gyro_z_bias)

        # invert sensor model to get altitude and airspeed
        self.estimated_state.altitude = self.lpf_abs.update(S.abs_pressure) / (MAV.rho * MAV.gravity)
        self.estimated_state.Va = np.sqrt(2 / MAV.rho * self.lpf_diff.update(S.diff_pressure))

        # estimate phi and theta with simple ekf
        self.attitude_ekf.update(S, self.estimated_state)  # estimated state being passed by reference here, no warning given :(

        # estimate pn, pe, Vg, chi, wn, we, psi
        self.position_ekf.true_state = self.true_state
        self.position_ekf.update(S, self.estimated_state)

        # not estimating these
        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state


class AlphaFilter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        self.y = self.alpha * self.y + (1 - self.alpha) * u
        return self.y


class EkfAttitude:
    # implement continous-discrete EKF to estimate roll and pitch angles
    def __init__(self):
        # x = [phi, theta]
        self.Q = np.array([[0.01, 0.],
                           [0., 0.003]])  # n x n where n is the state size
        self.Q_gyro = np.zeros((3,3))
        self.R_accel = np.array([[SENSOR.accel_sigma**2, 0, 0],
                                 [0, SENSOR.accel_sigma**2, 0],
                                 [0, 0, SENSOR.accel_sigma**2]]) * MAV.gravity  # m x m where m is sensor input size
        self.N = 10  # number of prediction step per sample
        self.xhat = np.array([0., 0.])  # initial state: phi, theta
        self.P = np.eye(2) * 0.
        self.Ts = SIM.ts_simulation / self.N
        self.gate_threshold = stats.chi2.isf(q=0.01, df=3)

    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.phi = self.xhat[0]
        state.theta = self.xhat[1]

    def f(self, x, measurement, state):
        # system dynamics for propagation model: xdot = f(x, u)
        phi = x[0]
        theta = x[1]
        p = state.p
        q = state.q
        r = state.r

        # G = np.array([[1, sin(phi)*tan(theta), cos(phi)*tan(theta), 0],
        #               [0, cos(phi), -sin(phi)]])
        f_ = np.array([p + q*sin(phi)*tan(theta)+r*cos(phi)*tan(theta),
                       q*cos(phi) - r*sin(phi)])
        return f_

    def h(self, x, measurement, state):
        # measurement model y
        phi = x[0]
        theta = x[1]
        g = MAV.gravity
        Va = state.Va
        p = state.p
        q = state.q
        r = state.r

        h_ = np.array([q*Va*sin(theta) + g*sin(theta),
                       r*Va*cos(theta) - p*Va*sin(theta) - g*cos(theta)*sin(phi),
                       -q*Va*cos(theta) - g*cos(theta)*cos(phi)])
        return h_

    def propagate_model(self, measurement, state):
        # model propagation
        for i in range(0, self.N):

            # propagate model
            self.xhat = self.xhat + self.Ts * self.f(self.xhat, measurement, state)
            # compute Jacobian
            q = state.q
            r = state.r
            phi, theta = self.xhat[0], self.xhat[1]
            A = np.array([[q*cos(phi)*tan(theta)-r*sin(phi)*tan(theta), (q*sin(phi)+r*cos(phi)) / (cos(theta)**2)],
                          [-q*sin(phi)-r*cos(phi), 0]])

            # compute G matrix for gyro noise
            G = 0
            # convert to discrete time models
            A_d = np.eye(2) + A * self.Ts + A @ A * self.Ts**2 / 2
            # update P with discrete time model
            self.P = A_d @ self.P @ A_d.T + self.Ts**2 * self.Q
            # self.P = self.P + self.Ts / self.N * (A @ self.P + self.P @ A.T + self.Q)

    def measurement_update(self, measurement, state):
        # measurement updates
        h = self.h(self.xhat, measurement, state)
        C = jacobian(self.h, self.xhat, measurement, state)
        y = np.array([measurement.accel_x, measurement.accel_y, measurement.accel_z]) * MAV.gravity
        S_inv = np.linalg.inv(self.R_accel + C @ self.P @ C.T)
        if (y-h).T @ S_inv @ (y-h) < self.gate_threshold:
            L = self.P @ C.T @ S_inv
            I = np.eye(2)
            self.P = (I - L @ C) @ self.P @ (I - L @ C).T + L @ self.R_accel @ L.T
            self.xhat = self.xhat + L @ (y - h)


class EkfPosition:
    # implement continous-discrete EKF to estimate pn, pe, Vg, chi, wn, we, psi
    def __init__(self):
        self.Q = np.array([[0.2, 0, 0, 0, 0, 0, 0.],
                           [0, 0.2, 0, 0, 0, 0, 0],
                           [0, 0, 10, 0, 0, 0, 0],
                           [0, 0, 0, 0.1, 0, 0, 0],
                           [0, 0, 0, 0, .1, 0, 0],
                           [0, 0, 0, 0, 0, 0.1, 0],
                           [0, 0, 0, 0, 0, 0, 30]])

        # 0.2 0.2 1 1.5 .1 .1 30

        # Assume all sensor data is independent, e.g. covariance is diagonal
        self.R_gps = np.array([[SENSOR.gps_n_sigma**2, 0, 0, 0],
                               [0, SENSOR.gps_e_sigma**2, 0, 0],
                               [0, 0, SENSOR.gps_h_sigma**2, 0],
                               [0, 0, 0, SENSOR.gps_course_sigma**2]])

        self.R_pseudo = np.array([[SENSOR.gps_Vg_sigma**2, 0],
                                  [0, SENSOR.gps_course_sigma**2]]) * 0.01
        self.N = 10  # number of prediction step per sample
        self.Ts = (SIM.ts_control / self.N)
        self.true_state = MsgState()
        self.xhat = np.array([0, 0, 10, 0, 0, 0, 0])  # pn, pe, Vg, chi, wn, we, psi
        self.P = np.zeros((7, 7))
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999
        self.pseudo_threshold = stats.chi2.isf(q=0.01, df=3)


    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.north = self.xhat.item(0)
        state.east = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, measurement, state):
        # system dynamics for propagation model: xdot = f(x, u)
        # State: [pn, pe, Vg, chi, wn, we, psi]

        # # Current Values:
        # Non-x values
        Va = state.Va
        phi = state.phi
        theta = state.theta
        q = state.q
        r = state.r
        g = MAV.gravity

        # x values
        pn = x[0]
        pe = x[1]
        Vg = x[2]
        chi = x[3]
        wn = x[4]
        we = x[5]
        psi = x[6]

        # Assumptions
        if abs(Vg) < 1e-3:
            Vg = 1e-3 * np.sign(Vg)

        # Equations of Motion:
        pn_dot = Vg*cos(chi)
        pe_dot = Vg*sin(chi)

        wn_dot = 0
        we_dot = 0

        chi_dot = g / Vg * tan(phi)
        psi_dot = q*sin(phi)/cos(theta) + r*cos(phi)/cos(theta)

        Vg_dot = (Va*psi_dot*(we*cos(psi)-wn*sin(psi))) / Vg

        f_ = np.array([pn_dot,
                       pe_dot,
                       Vg_dot,
                       chi_dot,
                       wn_dot,
                       we_dot,
                       psi_dot])
        return f_

    def A_f(self, x, measurement, state):
        # State: [pn, pe, Vg, chi, wn, we, psi]

        # # Current Values:
        # Non-x values
        Va = state.Va
        phi = state.phi
        theta = state.theta
        q = state.q
        r = state.r
        g = MAV.gravity

        # x values
        pn = x[0]
        pe = x[1]
        Vg = x[2]
        chi = x[3]
        wn = x[4]
        we = x[5]
        psi = x[6]

        # Assumptions
        if abs(Vg) < 1e-3:
            Vg = 1e-3 * np.sign(Vg)

        # Equations of Motion:
        pn_dot = Vg * cos(chi)
        pe_dot = Vg * sin(chi)

        wn_dot = 0
        we_dot = 0

        chi_dot = g / Vg * tan(phi)
        psi_dot = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)

        Vg_dot = (Va * psi_dot * (we * cos(psi) - wn * sin(psi))) / Vg

        A = np.array([[0, 0, cos(chi), -Vg*sin(chi), 0, 0, 0],
                      [0, 0, sin(chi), Vg*cos(chi), 0, 0, 0],
                      [0, 0, Vg_dot/Vg, 0, -psi_dot*Va*sin(psi)/Vg, psi_dot*Va*cos(psi)/Vg, Vg_dot],
                      [0, 0, -g*tan(phi)/Vg**2, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])

        return A

    def h_gps(self, x, measurement, state):
        # measurement model for gps measurements
        # State: [pn, pe, Vg, chi, wn, we, psi]
        pn = x[0]
        pe = x[1]
        Vg = x[2]
        chi = x[3]

        h_ = np.array([pn,
                       pe,
                       Vg,
                       chi])
        return h_

    def h_pseudo(self, x, measurement, state):
        # measurement model for wind triangle pseudo measurement
        # Non-x values
        Va = state.Va

        # x values
        # State: [pn, pe, Vg, chi, wn, we, psi]
        Vg = x[2]
        chi = x[3]
        wn = x[4]
        we = x[5]
        psi = x[6]

        ywn = Va * cos(psi) + wn - Vg * cos(chi)
        ywe = Va * sin(psi) + we - Vg * sin(chi)

        h_ = np.array([ywn,
                       ywe])
        return h_

    def C_gps(self, x, state):
        # C_gps * x = [pn, pe, Vg, chi]
        # State: [pn, pe, Vg, chi, wn, we, psi]

        return np.array([[1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]])

    def C_pseudo(self, x, state):
        # State: [pn, pe, Vg, chi, wn, we, psi]

        # Non-x values
        Va = state.Va

        # x values
        Vg = x[2]
        chi = x[3]
        psi = x[6]

        return np.array([[0, 0, -cos(chi), Vg*sin(chi), 1, 0, -Va*sin(psi)],
                         [0, 0, -sin(chi), -Vg*cos(chi), 0, 1, Va*cos(psi)]])

    def propagate_model(self, measurement, state):
        # model propagation
        for i in range(0, self.N):
            # propagate model
            self.xhat = self.xhat + self.Ts * self.f(self.xhat, measurement, state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, measurement, state)
            A = self.A_f(self.xhat, measurement, state)
            # convert to discrete time models
            A_d = np.eye(7) + A * self.Ts + A @ A * self.Ts**2 / 2
            # update P with discrete time model
            self.P = A_d @ self.P @ A_d.T + self.Ts**2 * self.Q

    def measurement_update(self, measurement, state):
        # always update based on wind triangle pseudo measurement
        h = self.h_pseudo(self.xhat, measurement, state)
        # C = jacobian(self.h_pseudo, self.xhat, measurement, state)
        C = self.C_pseudo(self.xhat, state)
        y = np.array([0, 0])
        S_inv = np.linalg.inv(self.R_pseudo + C @ self.P @ C.T)
        if True:  # (y-h).T @ S_inv @ (y-h) < self.pseudo_threshold:
            L = self.P @ C.T @ S_inv
            I = np.eye(7)
            self.P = (I - L @ C) @ self.P @ (I - L @ C).T + L @ self.R_pseudo @ L.T
            self.xhat = self.xhat + L @ (y - h)
        else:
            print("hit pseudo threshhold: ", (y-h).T @ S_inv @ (y-h))

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, measurement, state)
            # C = jacobian(self.h_gps, self.xhat, measurement, state)
            C = self.C_gps(self.xhat, state)
            y_chi = wrap(measurement.gps_course, h[3])
            y = np.array([measurement.gps_n,
                           measurement.gps_e,
                           measurement.gps_Vg,
                           y_chi])

            S_inv = np.linalg.inv(self.R_gps + C @ self.P @ C.T)
            L = self.P @ C.T @ S_inv
            I = np.eye(7)
            self.P = (I - L @ C) @ self.P @ (I - L @ C).T + L @ self.R_gps @ L.T
            self.xhat = self.xhat + L @ (y - h)

            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course


def jacobian(fun, x, measurement, state):
    # compute jacobian of fun with respect to x
    f = fun(x, measurement, state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.0001  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i] += eps
        f_eps = fun(x_eps, measurement, state)
        df = (f_eps - f) / eps
        J[:, i] = df[:]
    return J
