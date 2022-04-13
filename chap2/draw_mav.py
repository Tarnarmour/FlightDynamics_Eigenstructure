"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        4/15/2019 - BGM
"""
import numpy as np
import pyqtgraph.opengl as gl
from tools.rotations import Euler2Rotation


class DrawMav:
    def __init__(self, state, window):
        """
        Draw the MAV.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.north  # north position
            state.east  # east position
            state.altitude   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        # get points that define the non-rotated, non-translated mav and the mesh colors
        self.mav_points, self.mav_meshColors = self.get_points()

        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining mav
        rotated_points = self.rotate_points(self.mav_points, R)
        translated_points = self.translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        self.mav_body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                      vertexColors=self.mav_meshColors,  # defines mesh colors (Nx1)
                                      drawEdges=True,  # draw edges between mesh elements
                                      smooth=False,  # speeds up rendering
                                      computeNormals=False)  # speeds up rendering
        window.addItem(self.mav_body)  # add body to plot

    def update(self, state):
        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining mav
        rotated_points = self.rotate_points(self.mav_points, R)
        translated_points = self.translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        # draw MAV by resetting mesh using rotated and translated points
        self.mav_body.setMeshData(vertexes=mesh, vertexColors=self.mav_meshColors)

    def rotate_points(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = R @ points
        return rotated_points

    def translate_points(self, points, translation):
        "Translate points by the vector translation"
        translated_points = points + np.dot(translation, np.ones([1, points.shape[1]]))
        return translated_points

    def get_points(self):
        """"
            Points that define the mav, and the colors of the triangular mesh
            Define the points on the aircraft following diagram in Figure C.3
        """
        # define MAV body parameters
        unit_length = 0.25
        fuse_h = unit_length
        fuse_w = unit_length
        fuse_l1 = unit_length * 2
        fuse_l2 = unit_length
        fuse_l3 = unit_length * 4
        wing_l = unit_length
        wing_w = unit_length * 6
        tail_h = unit_length
        tail_l = unit_length
        tail_w = unit_length * 2

        # points are in NED coordinates
        # define the points on the aircraft following diagram Fig 2.14
        points = np.array([[fuse_l1, 0, 0],  # 1 [0]
                           [fuse_l2, fuse_w / 2, fuse_h / 2],  # 2 [1]
                           [fuse_l2, -fuse_w / 2, fuse_h / 2],  # 3 [2]
                           [fuse_l2, -fuse_w / 2, -fuse_h / 2],  # 4
                           [fuse_l2, fuse_w / 2, -fuse_h / 2],  # 5
                           [-fuse_l3, 0, 0],  # 6
                           [0, wing_w / 2, 0],  # 7
                           [wing_l, wing_w / 2, 0],  # 8
                           [wing_l, -wing_w / 2, 0],  # 9
                           [0, -wing_w / 2, 0],  # 10
                           [-fuse_l3 + tail_l, tail_w / 2, 0],  # 11
                           [-fuse_l3, tail_w / 2, 0],  # 12
                           [-fuse_l3, -tail_w / 2, 0],  # 13
                           [-fuse_l3 + tail_l, -tail_w / 2, 0],  # 14
                           [-fuse_l3 + tail_l, 0, 0],  # 15
                           [-fuse_l3, 0, -tail_h]  # 16
                           ]).T

        # scale points for better rendering
        scale = 50
        points = scale * points

        #   define the colors for each face of triangular mesh
        red = np.array([0.5, 0., 0., 1])
        green = np.array([0., 0.25, 0., 1])
        blue = np.array([0., 0., 0.5, 1])
        yellow = np.array([0.85, 0.3, 0., 1])
        purple = np.array([0.15, 0, 0.15, 1])
        gold = np.array([0.75, 0.5, 0.2, 1])
        meshColors = np.empty((13, 3, 4), dtype=np.float32)
        # Nose cone
        meshColors[0] = purple
        meshColors[1] = purple
        meshColors[2] = purple
        meshColors[3] = purple
        # Body
        meshColors[4] = purple
        meshColors[5] = purple
        meshColors[6] = purple
        meshColors[7] = purple
        # Tail horz
        meshColors[8] = gold
        meshColors[9] = gold
        # Tail vert
        meshColors[10] = gold
        # Wing
        meshColors[11] = gold
        meshColors[12] = gold

        return points, meshColors

    def points_to_mesh(self, points):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
          (a rectangle requires two triangular mesh faces)
        """
        points = points.T
        mesh = np.array([[points[0], points[1], points[2]],  # Nose cone .
                         [points[0], points[2], points[3]],  # .
                         [points[0], points[1], points[4]],  # .
                         [points[0], points[3], points[4]],  # .
                         [points[5], points[1], points[2]],  # Body .
                         [points[5], points[2], points[3]],  # .
                         [points[5], points[3], points[4]],  # .
                         [points[5], points[4], points[1]],  # .
                         [points[10], points[11], points[12]],  # Tail horz .
                         [points[10], points[13], points[12]],  # .
                         [points[5], points[15], points[14]],  # Tail vert .
                         [points[6], points[7], points[8]],  # Wing .
                         [points[6], points[9], points[8]]  # .
                         ])

        return mesh
