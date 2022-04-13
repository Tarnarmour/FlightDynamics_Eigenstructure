"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.rotations import Euler2Quaternion, Quaternion2Euler
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from message_types.msg_delta import MsgDelta

EPS = 0.001

def compute_model(mav, trim_state, trim_input):
    A_lon, B_lon, A_lat, B_lat, A, B = compute_ss_model(mav, trim_state, trim_input)
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    eig_lon, E = np.linalg.eig(A_lon)
    eig_lat, E = np.linalg.eig(A_lat)
    print(f"Longitudinal Eigenvalues: {eig_lon}\n\nLateal Eigenvalues: {eig_lat}")

    # write transfer function gains to file
    file = open('model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.elevator, trim_input.aileron, trim_input.rudder, trim_input.throttle))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
     A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
     A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
     A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
     A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon[0][0], B_lon[0][1],
     B_lon[1][0], B_lon[1][1],
     B_lon[2][0], B_lon[2][1],
     B_lon[3][0], B_lon[3][1],
     B_lon[4][0], B_lon[4][1],))
    file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
     A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
     A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
     A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
     A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat[0][0], B_lat[0][1],
     B_lat[1][0], B_lat[1][1],
     B_lat[2][0], B_lat[2][1],
     B_lat[3][0], B_lat[3][1],
     B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.write('A = np.array([\n    [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]])\n' %
               (A[0][0], A[0][1], A[0][2], A[0][3], A[0][4], A[0][5], A[0][6], A[0][7], A[0][8], A[0][9], A[0][10], A[0][11],
                A[1][0], A[1][1], A[1][2], A[1][3], A[1][4], A[1][5], A[1][6], A[1][7], A[1][8], A[1][9], A[1][10], A[1][11],
                A[2][0], A[2][1], A[2][2], A[2][3], A[2][4], A[2][5], A[2][6], A[2][7], A[2][8], A[2][9], A[2][10], A[2][11],
                A[3][0], A[3][1], A[3][2], A[3][3], A[3][4], A[3][5], A[3][6], A[3][7], A[3][8], A[3][9], A[3][10], A[3][11],
                A[4][0], A[4][1], A[4][2], A[4][3], A[4][4], A[4][5], A[4][6], A[4][7], A[4][8], A[4][9], A[4][10], A[4][11],
                A[5][0], A[5][1], A[5][2], A[5][3], A[5][4], A[5][5], A[5][6], A[5][7], A[5][8], A[5][9], A[5][10], A[5][11],
                A[6][0], A[6][1], A[6][2], A[6][3], A[6][4], A[6][5], A[6][6], A[6][7], A[6][8], A[6][9], A[6][10], A[6][11],
                A[7][0], A[7][1], A[7][2], A[7][3], A[7][4], A[7][5], A[7][6], A[7][7], A[7][8], A[7][9], A[7][10], A[7][11],
                A[8][0], A[8][1], A[8][2], A[8][3], A[8][4], A[8][5], A[8][6], A[8][7], A[8][8], A[8][9], A[8][10], A[8][11],
                A[9][0], A[9][1], A[9][2], A[9][3], A[9][4], A[9][5], A[9][6], A[9][7], A[9][8], A[9][9], A[9][10], A[9][11],
                A[10][0], A[10][1], A[10][2], A[10][3], A[10][4], A[10][5], A[10][6], A[10][7], A[10][8], A[10][9], A[10][10], A[10][11],
                A[11][0], A[11][1], A[11][2], A[11][3], A[11][4], A[11][5], A[11][6], A[11][7], A[11][8], A[11][9], A[11][10], A[11][11]))
    file.write('B = np.array([\n    [%f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f]])\n' %
               (B[0][0], B[0][1], B[0][2], B[0][3],
                B[1][0], B[1][1], B[1][2], B[1][3],
                B[2][0], B[2][1], B[2][2], B[2][3],
                B[3][0], B[3][1], B[3][2], B[3][3],
                B[4][0], B[4][1], B[4][2], B[4][3],
                B[5][0], B[5][1], B[5][2], B[5][3],
                B[6][0], B[6][1], B[6][2], B[6][3],
                B[7][0], B[7][1], B[7][2], B[7][3],
                B[8][0], B[8][1], B[8][2], B[8][3],
                B[9][0], B[9][1], B[9][2], B[9][3],
                B[10][0], B[10][1], B[10][2], B[10][3],
                B[11][0], B[11][1], B[11][2], B[11][3],))
    file.close()


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    mav._state = trim_state
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = Quaternion2Euler(trim_state[6:10])
    delta_e_trim = trim_input.elevator
    delta_a_trim = trim_input.aileron
    delta_r_trim = trim_input.rudder
    delta_t_trim = trim_input.throttle

    # define transfer function constants
    a_phi1 = -0.5 * MAV.rho * Va_trim**2 * MAV.S_wing * MAV.b * MAV.C_p_p * MAV.b / (2 * Va_trim)
    a_phi2 = 0.5 * MAV.rho * Va_trim**2 * MAV.S_wing * MAV.b * MAV.C_p_delta_a
    a_theta1 = -MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing / (2 * MAV.Jy) * MAV.C_m_q * MAV.c / (2 * Va_trim)
    a_theta2 = -MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing / (2 * MAV.Jy) * MAV.C_m_alpha
    a_theta3 = MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing / (2 * MAV.Jy) * MAV.C_m_delta_e

    # Compute transfer function coefficients using new propulsion model
    a_V1 = MAV.rho * Va_trim * MAV.S_wing / MAV.mass * (MAV.C_D_0 + MAV.C_D_alpha * alpha_trim + MAV.C_D_delta_e * delta_e_trim) - dT_dVa(mav, Va_trim, delta_t_trim) / MAV.mass
    a_V2 = dT_ddelta_t(mav, Va_trim, delta_t_trim) / MAV.mass
    a_V3 = MAV.gravity * np.cos(theta_trim - alpha_trim)

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3


def compute_ss_model(mav, trim_state, trim_input):
    x_euler = euler_state(trim_state)
    A = df_dx(mav, x_euler, trim_input)
    B = df_du(mav, x_euler, trim_input)
    # extract longitudinal states (u, w, q, theta, pd) and change pd to h
    # [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
    # [0   1   2   3  4  5  6    7      8    9  10 11]

    # [de, da, dr, dt]
    # [0,  1,  2,  3 ]

    A_lon = A[np.ix_([3, 5, 10, 7, 2], [3, 5, 10, 7, 2])]
    B_lon = B[np.ix_([3, 5, 10, 7, 2], [0, 3])]

    # extract lateral states (v, p, r, phi, psi)
    A_lat = A[np.ix_([4, 9, 11, 6, 8], [4, 9, 11, 6, 8])]
    B_lat = B[np.ix_([4, 9, 11, 6, 8], [1, 2])]

    # print(A)
    # print(B)

    return A_lon, B_lon, A_lat, B_lat, A, B

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    e = x_quat[6:10]
    rpy = np.array([Quaternion2Euler(e)]).T
    x_euler = np.vstack((x_quat[0:6], rpy, x_quat[10:None]))
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions
    rpy = x_euler[6:9, 0]
    e = Euler2Quaternion(rpy[0], rpy[1], rpy[2])
    x_quat = np.vstack((x_euler[0:6], e, x_euler[9:None]))
    return x_quat

def f_euler(mav, x_euler, delta):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state
    x_quat = quaternion_state(x_euler)
    mav._state = x_quat
    mav._update_velocity_data()
    f_and_m = mav._forces_moments(delta)
    f_quat_ = mav._derivatives(x_quat, f_and_m)
    f_quat_[2, 0] = -f_quat_[2, 0]  # compute hdot instead of d/dt p_down

    # should take partial quaternion2Euler w.r.t. input quaternion,
    # that gives a 3 x 4 matrix A_Q, then rpy = A_Q @ quaternion
    # NOT using Euler_state

    quat_0 = x_quat[6:10]
    A_Q = np.zeros((3, 4))
    for i in range(4):
        eps = EPS
        quat_eps = np.zeros_like(quat_0)
        quat_eps[i, 0] = eps
        quat_eps = quat_eps + quat_0
        A_Q[:, i] = ((np.array([Quaternion2Euler(quat_eps)]).T - np.array([Quaternion2Euler(quat_0)]).T) / eps)[:, 0]

    f_euler_ = np.vstack([f_quat_[0:6], A_Q @ f_quat_[6:10], f_quat_[10:None]])
    return f_euler_


def df_dx(mav, x_euler, delta):
    # take partial of f_euler with respect to x_euler
    A = np.zeros((12, 12))
    eps = EPS
    for i in range(12):
        x_eps = np.zeros_like(x_euler)
        x_eps[i, 0] = eps
        A[:, i] = ((f_euler(mav, x_euler + x_eps, delta) - f_euler(mav, x_euler, delta)) / eps)[:, 0]
    return A


def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to input
    B = np.zeros((12, 4))
    eps = EPS
    delta_eps = MsgDelta()
    for i in range(4):
        u_eps = np.zeros_like(delta.to_array())
        u_eps[i, 0] = eps
        delta_eps.from_array(delta.to_array() + u_eps)
        B[:, i] = ((f_euler(mav, x_euler, delta_eps) - f_euler(mav, x_euler, delta)) / eps)[:, 0]
    return B

def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = EPS
    T_eps, Q_eps = mav._motor_thrust_torque(Va + eps, delta_t)
    T, Q = mav._motor_thrust_torque(Va, delta_t)
    return (T_eps - T) / eps

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    eps = EPS
    T_eps, Q_eps = mav._motor_thrust_torque(Va, delta_t + eps)
    T, Q = mav._motor_thrust_torque(Va, delta_t)
    return (T_eps - T) / eps