import control as ct
import numpy as np
from numpy.linalg import norm, eig, svd
import matplotlib.pyplot as plt

from parameters import simulation_parameters as SIM
from parameters import aerosonde_parameters as MAV
from parameters import control_parameters as CONTROL

from chap4 import mav_dynamics, wind_simulation
from chap5 import compute_models, model_coef

np.set_printoptions(precision=3, linewidth=500, floatmode='fixed', suppress=True, sign=' ')

A_long = model_coef.A_lon
A_lat = model_coef.A_lat
B_long = model_coef.B_lon
B_lat = model_coef.B_lat
A = model_coef.A
B = model_coef.B

E, V = eig(A)
E_long, V_long = eig(A_long)
E_lat, V_lat = eig(A_lat)

print(f"State Space A Matrix:\n{A}\nState Space B Matrix:\n{B}\n")
print("States: [pn = north, pe = east, pd = down, u = vel x, v = vel y, w = vel z, phi = roll, theta = pitch, psi = yaw, p = roll rate, q = pitch rate, r = yaw rate")
print("Eigenvalues:\n", E)
print("Eigenvectors:")
for v in V.T: print(v)

print("\n\n")
print("Longitudinal Inputs:\n[de = elevator input, positive pushes tail up and nose down, dt = thrust input, positive increases speed]")
print("Longitudinal State:\n[u, w, q, theta, pd]\nu = north velocity, w = up velocity, q = body frame y angular velocity, theta = pitch angle, pd = -altitude]")
print(f"Longitudinal Eigenvalues:\n{E_long}")
print(f"Longitudinal Eigenvectors:")
for v in V_long.T: print(v)

print("\n\n")
print("Lateral Inputs:\n[da = aileron, dr = rudder input]")
print("Lateral State:\n[v, p, r, phi, psi]\nv = east velocity, p = roll rate, r = yaw rate, phi = roll, psi = yaw")
print(f"Lateral Eigenvalues:\n{E_lat}")
print(f"Lateral Eigenvectors:")
for v in V_lat.T: print(v)
