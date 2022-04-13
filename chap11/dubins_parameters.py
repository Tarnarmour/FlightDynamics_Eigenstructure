# dubins_parameters
#   - Dubins parameters that define path between two configurations
#
# mavsim_matlab 
#     - Beard & McLain, PUP, 2012
#     - Update history:  
#         3/26/2019 - RWB
#         4/2/2020 - RWB

import numpy as np
import sys
from numpy import sin, cos, tan, sqrt, pi
from numpy.linalg import norm
sys.path.append('..')


class DubinsParameters:
    def __init__(self, ps=9999*np.ones((3,)), chis=9999,
                 pe=9999*np.ones((3,)), chie=9999, R=9999):
        if R == 9999:
            L = R
            cs = ps
            lams = R
            ce = ps
            lame = R
            w1 = ps
            q1 = ps
            w2 = ps
            w3 = ps
            q3 = ps
        else:
            L, cs, lams, ce, lame, w1, q1, w2, w3, q3 \
                = compute_parameters(ps, chis, pe, chie, R)
        self.p_s = ps
        self.chi_s = chis
        self.p_e = pe
        self.chi_e = chie
        self.radius = R
        self.length = L
        self.center_s = cs
        self.dir_s = lams
        self.center_e = ce
        self.dir_e = lame
        self.r1 = w1
        self.n1 = q1
        self.r2 = w2
        self.r3 = w3
        self.n3 = q3

    def update(self, ps, chis, pe, chie, R):
        L, cs, lams, ce, lame, w1, q1, w2, w3, q3 \
            = compute_parameters(ps, chis, pe, chie, R)
        self.p_s = ps
        self.chi_s = chis
        self.p_e = pe
        self.chi_e = chie
        self.radius = R
        self.length = L
        self.center_s = cs
        self.dir_s = lams
        self.center_e = ce
        self.dir_e = lame
        self.r1 = w1
        self.n1 = q1
        self.r2 = w2
        self.r3 = w3
        self.n3 = q3


def compute_parameters(ps, chis, pe, chie, R):
    pass
    q = pe - ps
    ell = norm(q)
    if ell < 2 * R:
        print('Error in Dubins Parameters: The distance between nodes must be larger than 2R.')
    else:
        # compute start and end circles
        crs = ps + R * rotz(pi/2) @ np.array([cos(chis), sin(chis), 0])
        cls = ps + R * rotz(-pi/2) @ np.array([cos(chis), sin(chis), 0])
        cre = pe + R * rotz(pi/2) @ np.array([cos(chie), sin(chie), 0])
        cle = pe + R * rotz(-pi/2) @ np.array([cos(chie), sin(chie), 0])

        # compute L1
        v = cre - crs
        phi = np.arctan2(v[1], v[0])
        L1 = norm(crs - cre) + R * (mod(2*pi + mod(phi - pi/2) - mod(chis - pi/2)) + mod(2*pi + mod(chie - pi/2) - mod(phi - pi/2)))

        # compute L2
        v = cle - crs
        ell = norm(v)
        theta = np.arctan2(v[1], v[0])
        theta2 = theta - pi/2 + np.arcsin(2*R/ell)
        if np.isnan(theta2):
            print('error here 1')
        if not np.isreal(theta2):
            L2 = np.inf
        else:
            L2 = sqrt(ell**2 - 4*R**2) + R * (mod(2*pi + mod(theta2) - mod(chis-pi/2)) + mod(2*pi + mod(theta2+pi) - mod(chie+pi/2)))

        # compute L3
        v = cre - cls
        ell = norm(v)
        theta = np.arctan2(v[1], v[0])
        theta2 = np.arccos(2*R / ell)
        if np.isnan(theta2):
            print('error here 2')
        if not np.isreal(theta2):
            L3 = np.inf
        else:
            L3 = sqrt(ell**2 - 4*R**2) + R * (mod(2*pi + mod(chis+pi/2) - mod(theta+theta2)) + mod(2*pi + mod(chie-pi/2) - mod(theta+theta2-pi)))

        # compute L4
        v = cle - cls
        theta = np.arctan2(v[1], v[0])
        L4 = norm(v) + R*(mod(2*pi + mod(chis+pi/2) - mod(theta+pi/2)) + mod(2*pi + mod(theta+pi/2) - mod(chie+pi/2)))
        # L is the minimum distance
        L = np.min([L1, L2, L3, L4])
        idx = np.argmin([L1, L2, L3, L4])

        # [start circle, start dir, end circle, end dir, z1]
        # lams
        if idx == 0:
            cs = crs
            lams = 1
            ce = cre
            lame = 1
            q1 = (cre - crs) / norm(cre - crs)
            w1 = cs + R * rotz(-pi/2) @ q1
            w2 = ce + R * rotz(-pi/2) @ q1
            # q1 = (w2 - w1) / norm(w2 - w1)
            # print("Dubins Path 1")
        elif idx == 1:
            cs = crs
            lams = 1
            ce = cle
            lame = -1
            v = cle - crs
            ell = norm(v)
            theta = np.arctan2(v[1], v[0])
            theta2 = theta - pi/2 + np.arcsin(2*R/ell)
            q1 = rotz(theta2 + pi/2)[:, 0]
            w1 = cs + R * rotz(theta2)[:, 0]
            w2 = ce + R * rotz(theta2 + pi)[:, 0]
            # q1 = (w2 - w1) / norm(w2 - w1)
            # print("Dubins Path 2")
        elif idx == 2:
            cs = cls
            lams = -1
            ce = cre
            lame = 1
            v = cre - cls
            ell = norm(v)
            theta = np.arctan2(v[1], v[0])
            theta2 = np.arccos(2 * R / ell)
            q1 = rotz(theta + theta2 - pi/2)[:, 0]
            w1 = cs + R * rotz(theta + theta2)[:, 0]
            w2 = ce + R * rotz(theta + theta2 - pi)[:, 0]
            # q1 = (w2 - w1) / norm(w2 - w1)
            # print("Dubins Path 3")
        else:  # idx == 3:
            cs = cls
            lams = -1
            ce = cle
            lame = -1
            v = cle - cls
            theta = np.arctan2(v[1], v[0])
            q1 = (ce - cs) / norm(ce - cs)
            w1 = cs + R * rotz(pi/2) @ q1
            w2 = ce + R * rotz(pi/2) @ q1
            # q1 = (w2 - w1) / norm(w2 - w1)
            # print("Dubins Path 4")
        w3 = pe
        q3 = rotz(chie)[:, 0]

        return L, cs, lams, ce, lame, w1, q1, w2, w3, q3


def rotz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])


def mod(x):
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x


