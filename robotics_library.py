# import libraries
import numpy as np
import pandas as pd


# goals


class Robotic_Manipulator_Naive(object):

    def __init__(self, link_lengthes):
        """
        rotation axises and relationship between consecutive links are pre-determined and cannot be changed
        :param link_lengthes:
        """
        self.link_lengthes = link_lengthes
        self.joint_relative_locations = np.asarray([[0, 0, self.link_lengthes[0], 1],
                                                    [0, 0, self.link_lengthes[1], 1],
                                                    [0, self.link_lengthes[2], 0, 1],
                                                    [0, self.link_lengthes[3], 0, 1],
                                                    [0, self.link_lengthes[4], 0, 1],
                                                    [self.link_lengthes[5], 0, 0, 1],
                                                    [self.link_lengthes[6], 0, 0, 1]
                                                    ])
        self.rotation_axises = ["z", "x", "x", "z", "y", "y"]
        self.initial_relative_angles = [0, 0, 0, 0, np.pi, 0]

        def r10(q, idx=0):
            initial_q = self.initial_relative_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), -np.sin(q), 0, 0],
                             [np.sin(q), np.cos(q), 0, 0],
                             [0, 0, 1, l],
                             [0, 0, 0, 1]])
            return(hm)

        def r21(q, idx=1):
            initial_q = self.initial_relative_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[1, 0, 0, 0],
                             [0, np.cos(q), -np.sin(q), 0],
                             [0, np.sin(q), np.cos(q), l],
                             [0, 0, 0, 1]])
            return(hm)

        def r32(q, idx=2):
            initial_q = self.initial_relative_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[1, 0, 0, 0],
                             [0, np.cos(q), -np.sin(q), 0],
                             [0, np.sin(q), np.cos(q), l],
                             [0, 0, 0, 1]])
            return(hm)

        def r43(q, idx=3):
            initial_q = self.initial_relative_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), -np.sin(q), 0, 0],
                             [np.sin(q), np.cos(q), 0, l],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
            return(hm)

        def r54(q, idx=4):
            initial_q = self.initial_relative_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), 0, -np.sin(q), 0],
                             [0, 1, 0, l],
                             [np.sin(q), 0, np.cos(q), 0],
                             [0, 0, 0, 1]])
            return(hm)

        def r65(q, idx=5):
            initial_q = self.initial_relative_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), 0, -np.sin(q), l],
                             [0, 1, 0, 0],
                             [np.sin(q), 0, np.cos(q), 0],
                             [0, 0, 0, 1]])
            return(hm)


        self.ht_list = [r10, r21, r32, r43, r54, r65]


    def forward_kinematics(self, qs, x, reference_frame=6):
        transform_list = self.ht_list[:reference_frame]
        transform_list = transform_list[::-1]

        qs = qs[:reference_frame]
        qs = qs[::-1]

        new_x = x
        for hm, q in zip(transform_list, qs):
            new_x = np.dot(hm(q), new_x)
        return(new_x)

    def loc_joints(self, qs):
        joint_abs_locations = []
        for idx, rel_loc in enumerate(self.joint_relative_locations):
            tmp_loc = self.forward_kinematics(qs, rel_loc, reference_frame=idx)
            joint_abs_locations.append(tmp_loc)
        return(np.asarray(joint_abs_locations))




### test

link_lengthes = [1, 10, 1, 1, 1, 1, 1]
rm = Robotic_Manipulator_Naive(link_lengthes)

test_x = np.asarray([0, 0, 1, 1])
test_x = np.asarray([1, 0, 2, 1])

# test with one single point
qs = [0, 0, np.pi/6, np.pi/6, np.pi/6, np.pi/6]
qs[0] = np.pi/6
new_x = rm.forward_kinematics(qs, test_x, reference_frame=1)
print new_x

# test the position of each joint
qs = [0] * 6
qs[0] = np.pi / 6
joint_locs = rm.loc_joints(qs)
print joint_locs





def homogeneous_transformation():
    pass




def rotation_matrix(q, axis):
    if axis == "z":
        rm = np.asarray([[np.cos(q), np.sin(q), 0],
                         [-np.sin(q), np.cos(q), 0],
                         [0, 0, 1]]).T
    elif axis == "y":
        rm = np.asarray([[np.cos(q), 0, np.sin(q)],
                         [0, 1, 0],
                         [-np.sin(q), 0, np.cos(q)]]).T
    elif axis == "x":
        rm = np.asarray([[1, 0, 0],
                         [0, np.cos(q), np.sin(q)],
                         [0, -np.sin(q), np.cos(q)]]).T
    return(rm)



class RobotManipulator(object):

    def __init__(self, link_lengths, rotation_axises):
        # how to represent the link and rotation angles
        # assumption: 1.
        self.links = link_lengths
        self.axises = rotation_axises
        self.num_joints = len(rotation_axises)
        pass

    def forward_kinematics(self, qs):
        pass

    def inverse_kinematics(self, ee_loc):
        pass


