import sys
sys.path.append('..')
import numpy as np

######################################################################################
                #   sample times, etc
######################################################################################
ts_simulation = 0.005  # smallest time step for simulation
start_time = 0.  # start time for simulation
end_time = 10.  # end time for simulation

data_sample_period = 1
data_plotting_period = 10
drawing_update_period = 1

ts_video = 0.015  # write rate for video

ts_control = ts_simulation  # sample rate for the controller


