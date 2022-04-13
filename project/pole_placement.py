import control as ct
import numpy as np
from numpy.linalg import norm, eig, svd
import matplotlib.pyplot as plt

from parameters import simulation_parameters as SIM
from parameters import aerosonde_parameters as MAV
from parameters import control_parameters as CONTROL
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta

from chap4.mav_dynamics import MavDynamics
from chap4.wind_simulation import WindSimulation
from chap5 import compute_models, model_coef
from chap5.trim import compute_trim

"""Longitudinal State:\n[u, w, q, theta, pd]\nu = north velocity, w = up velocity, q = body frame y angular velocity, theta = pitch angle, pd = -altitude]"""
A_long = model_coef.A_lon
B_long = model_coef.B_lon
Cr_long = np.array([[1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1]])

"""Lateral State:\n[v, p, r, phi, psi]\nv = east velocity, p = roll rate, r = yaw rate, phi = roll, psi = yaw"""
A_lat = model_coef.A_lat
B_lat = model_coef.B_lat
Cr_lat = np.array([[0, 0, 0, 0, 1],
                   [1, 0, 0, 0, 0]])

open_long_poles = [ 0.0000+0.0000j, -4.8778+9.8690j, -4.8778-9.8690j, -0.1175+0.4925j, -0.1175-0.4925j]
des_long_poles = [ 0.0000+0.0000j, -4.8778+9.8690j, -4.8778-9.8690j, -0.1175+0.4925j, -0.1175-0.4925j]
K_long = np.squeeze(np.asarray(ct.place(A_long, B_long, des_long_poles)))
kr_long = -1 * np.linalg.inv(Cr_long @ np.linalg.inv(A_long - B_long @ K_long) @ B_long)

open_lat_poles = [-22.4411+0.0000j,  -1.1402+4.6549j,  -1.1402-4.6549j,   0.1021+0.0000j,   0.0113+0.0000j]
des_lat_poles = [-22.4411+0.0000j,  -1.1402+4.6549j,  -1.1402-4.6549j,   -0.1021+0.0000j,   -0.0113+0.0000j]
K_lat = np.squeeze(np.asarray(ct.place(A_lat, B_lat, des_lat_poles)))
kr_lat = -1 * np.linalg.inv(Cr_lat @ np.linalg.inv(A_lat - B_lat @ K_lat) @ B_lat)

print(f"Longitudinal Control:\nK_long:\n{K_long}\nkr_long:\n{kr_long}\nLateral Control:\nK_lat:\n{K_lat}\nkr_lat:\n{kr_lat}")
print("\n\n")
