"""
mavsim_python
    - Chapter 11 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        3/26/2019 - RWB
        2/27/2020 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import copy
import parameters.simulation_parameters as SIM
import parameters.planner_parameters as PLAN

from chap3.data_viewer import DataViewer
from chap4.wind_simulation import WindSimulation
from chap6.autopilot import Autopilot
from chap7.mav_dynamics import MavDynamics
from chap8.observer import Observer
from chap10.path_follower import PathFollower
from chap11.path_manager import PathManager
from chap11.waypoint_viewer import WaypointViewer

# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
do_data = False
waypoint_view = WaypointViewer()  # initialize the viewer
if do_data:
   data_view = DataViewer()  # initialize view of data plots
if VIDEO is True:
    from chap2.video_writer import VideoWriter
    video = VideoWriter(video_name="chap11_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)
autopilot = Autopilot(SIM.ts_simulation)
initial_state = copy.deepcopy(mav.true_state)
observer = Observer(SIM.ts_simulation, initial_state)
path_follower = PathFollower()
path_manager = PathManager()

# waypoint definition
from message_types.msg_waypoints import MsgWaypoints
waypoints = MsgWaypoints()
# waypoints.type = 'straight_line'
# waypoints.type = 'fillet'
waypoints.type = 'dubins'
Va = PLAN.Va0
L = 500
waypoints.add(np.array([0, 0, -100]), Va, np.radians(0), np.inf, 0, 0)
waypoints.add(np.array([L, 0, -100]), Va, np.radians(135), np.inf, 0, 0)
waypoints.add(np.array([0, L, -100]), Va, np.radians(0), np.inf, 0, 0)
waypoints.add(np.array([L, L, -100]), Va, np.radians(-135), np.inf, 0, 0)

# waypoints.add(np.array([0, 0, -100]), Va, np.radians(0), np.inf, 0, 0)
# waypoints.add(np.array([1000, 0, -100]), Va, np.radians(-135), np.inf, 0, 0)
# waypoints.add(np.array([0, -1000, -100]), Va, np.radians(135), np.inf, 0, 0)
# waypoints.add(np.array([-1000, 0, -100]), Va, np.radians(0), np.inf, 0, 0)

"""
add state assuming that we are in the wrong side of half plane at the start of the circles
"""

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
flag = 0
while sim_time < 500:
    # -------observer-------------
    measurements = mav.sensors()  # get sensor measurements
    estimated_state = observer.update(measurements)  # estimate states from measurements
    true_state = mav.true_state
    state = true_state


    # -------path manager-------------
    # wrap around to keep going in circuit
    if path_manager.ptr_current == 2:
        path_manager.ptr_previous = -3
        path_manager.ptr_current = -2
        path_manager.ptr_next = -1
    path = path_manager.update(waypoints, PLAN.R_min, state)

    # -------path follower-------------
    autopilot_commands = path_follower.update(path, state)

    # -------autopilot-------------
    delta, commanded_state = autopilot.update(autopilot_commands, state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    # print(current_wind)
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    waypoint_view.update(mav.true_state, path, waypoints)  # plot path and MAV

    if do_data:
        data_view.update(mav.true_state,  # true states
                         estimated_state,  # estimated states
                         commanded_state,  # commanded states
                         delta,  # input to aircraft
                         SIM.ts_plotting)

    if VIDEO is True:
        video.update(sim_time)

    # -------increment time-------------
    sim_time += SIM.ts_simulation

if VIDEO is True:
    video.close()




