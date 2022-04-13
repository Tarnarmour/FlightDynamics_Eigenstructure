"""
msg_waypoints
    - messages type for input to path manager
    
part of mavsim_python
    - Beard & McLain, PUP, 2012
    - Last update:
        3/26/2019 - RWB
        3/31/2020 - RWB
"""
import numpy as np


class MsgWaypoints:
    def __init__(self):
        # the first two flags are used for interacting with the path planner
        #
        # flag to indicate waypoints recently changed (set by planner)
        self.flag_waypoints_changed = True
        self.plot_updated = False  # used to plot waypoints

        # type of waypoint following:
        #   - straight line following
        #   - fillets between straight lines
        #   - follow dubins paths
        self.type = 'straight_line'
        # self.type = 'fillet'
        # self.type = 'dubins'
        # current number of valid waypoints in memory
        self.num_waypoints = 0
        # [n, e, d] - coordinates of waypoints

        # # NOTE! I changed these to be lists because I want to use 1-D arrays
        self.ned = []
        # the airspeed that is commanded along the waypoints
        self.airspeed = []
        # the desired course at each waypoint (used only for Dubins paths)
        self.course = []

        # these last three variables are used by the path planner running cost at each node
        self.cost = np.array([])
        # index of the parent to the node
        self.parent = []
        # can this node connect to the goal?
        self.connect_to_goal = np.array([])
        # Do orbit idle?
        self.idle = False

    def add(self, ned=np.array([0, 0, 0]), airspeed=0,
            course=np.inf, cost=0, parent=0, connect_to_goal=0):
        self.num_waypoints = self.num_waypoints + 1
        self.ned.append(ned)
        self.airspeed.append(airspeed)
        self.course.append(course)

        self.cost = np.append(self.cost, cost)
        self.parent.append(int(parent))
        self.connect_to_goal = np.append(self.connect_to_goal, connect_to_goal)