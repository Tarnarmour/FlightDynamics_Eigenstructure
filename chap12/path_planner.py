# path planner for mavsim_python
#
# mavsim_python
#     - Beard & McLain, PUP, 2012
#     - Last updated:
#         4/3/2019 - BGM
import numpy as np
import sys

sys.path.append('..')
from message_types.msg_waypoints import MsgWaypoints
from parameters.aerosonde_parameters import Va0
from chap12.rrt_straight_line import RRTStraightLine
from chap12.rrt_dubins import RRTDubins


class PathPlanner:
    def __init__(self):
        # waypoints definition
        self.waypoints = MsgWaypoints()
        self.rrt_straight_line = RRTStraightLine()
        self.rrt_dubins = RRTDubins()

    def update(self, world_map, state, radius, target=None):
        print('planning...')
        # this flag is set for one time step to signal a redraw in the viewer
        # planner_flag = 'simple_straight'  # return simple waypoint path
        # planner_flag = 'simple_dubins'  # return simple dubins waypoint path
        # planner_flag = 'rrt_straight'  # plan path through city using straight-line RRT
        planner_flag = 'rrt_dubins'  # plan path through city using dubins RRT
        if planner_flag == 'simple_straight':
            Va = Va0
            self.waypoints.type = 'fillet'
            self.waypoints.add(np.array([0, 0, -100]), Va, np.inf, np.inf, 0, 0)
            self.waypoints.add(np.array([500, 0, -100]), Va, np.inf, np.inf, 0, 0)
            self.waypoints.add(np.array([0, 500, -100]), Va, np.inf, np.inf, 0, 0)
            self.waypoints.add(np.array([500, 500, -100]), Va, np.inf, np.inf, 0, 0)

        elif planner_flag == 'simple_dubins':
            Va = Va0
            self.waypoints.type = 'dubins'
            self.waypoints.add(np.array([0, 0, -100]), Va, np.radians(0), np.inf, 0, 0)
            self.waypoints.add(np.array([1000, 0, -100]), Va, np.radians(135), np.inf, 0, 0)
            self.waypoints.add(np.array([0, 1000, -100]), Va, np.radians(0), np.inf, 0, 0)
            self.waypoints.add(np.array([1000, 1000, -100]), Va, np.radians(-135), np.inf, 0, 0)

        elif planner_flag == 'rrt_straight':
            desired_airspeed = Va0
            desired_altitude = 100
            # start pose is current pose
            start_pose = np.array([state.north, state.east, -state.altitude, state.chi])
            # desired end pose
            if target is None:
                if np.linalg.norm(start_pose[0:2]) < world_map.city_width / 2:
                    end_pose = np.array([world_map.city_width, world_map.city_width,
                                         -desired_altitude, state.chi])
                else:  # or to the bottom-left corner of world_map
                    end_pose = np.array([0, 0, -desired_altitude, state.chi])
            else:
                end_pose = target

            if np.linalg.norm(start_pose[0:3] - end_pose[0:3]) < radius:
                self.waypoints = MsgWaypoints()
                self.waypoints.idle = True
            self.waypoints = self.rrt_straight_line.update(start_pose, end_pose,
                                                           desired_airspeed, world_map, radius)
            self.waypoints.type = 'fillet'

        elif planner_flag == 'rrt_dubins':
            desired_airspeed = Va0
            desired_altitude = 100
            # start pose is current pose
            start_pose = np.array([state.north, state.east,
                                   -desired_altitude, state.chi])
            # desired end pose
            # either plan to the top-right corner of world_map
            if target is None:
                if np.linalg.norm(start_pose[0:2]) < world_map.city_width / 2:
                    end_pose = np.array([world_map.city_width, world_map.city_width,
                                         -desired_altitude, state.chi])
                else:  # or to the bottom-left corner of world_map
                    end_pose = np.array([0, 0, -desired_altitude, state.chi])
            else:
                end_pose = target
            self.waypoints = self.rrt_dubins.update(start_pose, end_pose,
                                                    desired_airspeed, world_map, radius)
            self.waypoints.type = 'dubins'
        else:
            print("Error in Path Planner: Undefined planner type.")
        self.waypoints.plot_updated = False
        print('...done planning.')
        return self.waypoints
