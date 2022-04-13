"""
mavsim_python
    - Chapter 8 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/21/2019 - RWB
        2/24/2020 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import copy
import parameters.simulation_parameters as SIM

from chap2.mav_viewer import MavViewer
from chap3.data_viewer import DataViewer
from chap4.wind_simulation import WindSimulation
from chap6.autopilot import Autopilot
from chap7.mav_dynamics import MavDynamics
from chap8.observer import Observer
#from chap8.observer_full import Observer
from tools.signals import Signals

# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
mav_view = MavViewer()  # initialize the mav viewer
data_view = DataViewer()  # initialize view of data plots
if VIDEO is True:
    from chap2.video_writer import VideoWriter
    video = VideoWriter(video_name="chap8_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)
autopilot = Autopilot(SIM.ts_simulation)
initial_state = copy.deepcopy(mav.true_state)
initial_measurements = copy.deepcopy(mav.sensors())
observer = Observer(SIM.ts_simulation, initial_state, initial_measurements)



# autopilot commands
from message_types.msg_autopilot import MsgAutopilot
from message_types.msg_state import MsgState
from tools.john_custom import CopyState
commands = MsgAutopilot()
Va_command = Signals(dc_offset=25.0,
                     amplitude=3.0, # 3
                     start_time=2.0,
                     frequency = 0.11)
h_command = Signals(dc_offset=100.0,
                    amplitude=0,  # 10
                    start_time=1.0,
                    frequency=0.2)
chi_command = Signals(dc_offset=np.radians(0),  # 180
                      amplitude=np.radians(15),  # 45
                      start_time=1.0,
                      frequency=0.07)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < 20.:

    # -------autopilot commands-------------
    commands.airspeed_command = Va_command.square(sim_time)
    commands.course_command = chi_command.square(sim_time)
    commands.altitude_command = h_command.step(sim_time)

    # -------autopilot-------------
    measurements = mav.sensors()  # get sensor measurements
    true_state = mav.true_state
    observer.true_state = true_state
    estimated_state = observer.update(measurements)  # estimate states from measurements
    delta, commanded_state = autopilot.update(commands, estimated_state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    mav_view.update(mav.true_state)  # plot body of MAV
    data_view.update(mav.true_state,  # true states
                     estimated_state,  # estimated states
                     commanded_state,  # commanded states
                     delta,  # input to aircraft
                     SIM.ts_simulation)
    if VIDEO is True:
        video.update(sim_time)

    # -------increment time-------------
    sim_time += SIM.ts_simulation


if VIDEO is True:
    video.close()




