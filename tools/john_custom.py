from message_types.msg_state import MsgState

def CopyState(in_state=MsgState()):
    out_state = MsgState()
    out_state.north = in_state.north  # inertial north position in meters
    out_state.east = in_state.east  # inertial east position in meters
    out_state.altitude = in_state.altitude  # inertial altitude in meters
    out_state.phi = in_state.phi  # roll angle in radians
    out_state.theta = in_state.theta  # pitch angle in radians
    out_state.psi = in_state.psi  # yaw angle in radians
    out_state.Va = in_state.Va  # airspeed in meters/sec
    out_state.alpha = in_state.alpha  # angle of attack in radians
    out_state.beta = in_state.beta  # sideslip angle in radians
    out_state.p = in_state.p  # roll rate in radians/sec
    out_state.q = in_state.q  # pitch rate in radians/sec
    out_state.r = in_state.r  # yaw rate in radians/sec
    out_state.Vg = in_state.Vg  # groundspeed in meters/sec
    out_state.gamma = in_state.gamma  # flight path angle in radians
    out_state.chi = in_state.chi  # course angle in radians
    out_state.wn = in_state.wn  # inertial windspeed in north direction in meters/sec
    out_state.we = in_state.we  # inertial windspeed in east direction in meters/sec
    out_state.bx = in_state.bx  # gyro bias along roll axis in radians/sec
    out_state.by = in_state.by  # gyro bias along pitch axis in radians/sec
    out_state.bz = in_state.bz  # gyro bias along yaw axis in radians/sec
    return out_state