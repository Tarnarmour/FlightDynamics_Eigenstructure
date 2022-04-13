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
from tools.rotations import Quaternion2Euler

from pole_placement import K_lat, K_long, kr_long, kr_lat


def q2E(state):
    x = np.zeros((12,))
    x[0:6] = state[0:6]
    x[6:9] = Quaternion2Euler(state[6:10])
    x[9:None] = state[10:None]
    return x


mav = MavDynamics(SIM.ts_simulation)
wind_sim = WindSimulation(SIM.ts_simulation)

long_ix = [3, 5, 10, 7, 2]
lat_ix = [4, 9, 11, 6, 8]

Va = 25
gamma = 0.
trim_state_q, trim_input = model_coef.x_trim, model_coef.u_trim
trim_state = q2E(np.squeeze(trim_state_q))
x_eq = np.squeeze(trim_state)
u_eq = np.squeeze(trim_input)
delta = MsgDelta()

disturbance = np.zeros_like(trim_state)
disturbance[3] = 10.0
mav.external_set_state(trim_state_q + disturbance)
state_q = np.squeeze(mav._state)
state = q2E(state_q)

t = 0.0
t_end = 25.
dt = SIM.ts_simulation
ts = np.array([t])
state_hist = np.squeeze(trim_state)

r_psi = 5.0 * np.pi / 180
r_u = 25
r_pd = 0.0

r_long = np.array([r_u, r_pd])
r_lat = np.array([r_psi, 0])

fig, ax = plt.subplots()

while t < t_end:

    x_long = state[long_ix] - x_eq[long_ix]
    x_lat = state[lat_ix] - x_eq[lat_ix]

    u_long = -K_long @ x_long + kr_long @ r_long * 0.0
    u_lat = -K_lat @ x_lat + kr_lat @ r_lat * 0.0
    u = np.array([u_long[0], u_lat[0], u_lat[1], u_long[1]])
    # print(u)

    delta.from_array(u + u_eq)
    wind = wind_sim.update()
    mav.update(delta, wind)
    state_q = np.squeeze(mav._state)
    state = q2E(state_q)

    t += dt
    ts = np.append(ts, t)
    state_hist = np.vstack((state_hist, state))

    ax.plot(ts, state_hist[:, 8] * 180 / np.pi)
    ax.plot(ts, state_hist[:, 3])

    plt.pause(0.001)

"""States: pn = north, pe = east, pd = down, u = vel x, v = vel y, w = vel z, phi = roll, theta = pitch, psi = yaw, p = roll rate, q = pitch rate, r = yaw rate"""
"""            0            1        2          3            4         5           6            7           8           9              10               11"""
# ax.plot(ts, state_hist[:, 0])
ax.plot(ts, state_hist[:, 8] * 180 / np.pi)
ax.plot(ts, state_hist[:, 3])

plt.legend(["yaw angle", "x velocity"])

plt.pause(0.1)
plt.show()


