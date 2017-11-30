# import libraries
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers

import tensorflow as tf


class Robotic_Manipulator_Naive(object):
    def __init__(self, link_lengthes, initial_angles):
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
        self.initial_relative_angles = initial_angles

        def r10(q, idx=0):
            initial_q = self.initial_relative_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), -np.sin(q), 0, 0],
                             [np.sin(q), np.cos(q), 0, 0],
                             [0, 0, 1, l],
                             [0, 0, 0, 1]])
            return (hm)

        def r21(q, idx=1):
            initial_q = self.initial_relative_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[1, 0, 0, 0],
                             [0, np.cos(q), -np.sin(q), 0],
                             [0, np.sin(q), np.cos(q), l],
                             [0, 0, 0, 1]])
            return (hm)

        def r32(q, idx=2):
            initial_q = self.initial_relative_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[1, 0, 0, 0],
                             [0, np.cos(q), -np.sin(q), l],
                             [0, np.sin(q), np.cos(q), 0],
                             [0, 0, 0, 1]])
            return (hm)

        def r43(q, idx=3):
            initial_q = self.initial_relative_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), -np.sin(q), 0, 0],
                             [np.sin(q), np.cos(q), 0, l],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
            return (hm)

        def r54(q, idx=4):
            initial_q = self.initial_relative_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), 0, -np.sin(q), 0],
                             [0, 1, 0, l],
                             [np.sin(q), 0, np.cos(q), 0],
                             [0, 0, 0, 1]])
            return (hm)

        def r65(q, idx=5):
            initial_q = self.initial_relative_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), 0, -np.sin(q), l],
                             [0, 1, 0, 0],
                             [np.sin(q), 0, np.cos(q), 0],
                             [0, 0, 0, 1]])
            return (hm)

        self.ht_list = [r10, r21, r32, r43, r54, r65]
        self.joint_abs_locations = self.loc_joints(qs=[0] * len(self.rotation_axises))

    def forward_kinematics(self, qs, x, reference_frame=6):
        transform_list = self.ht_list[:reference_frame]
        transform_list = transform_list[::-1]

        qs = qs[:reference_frame]
        qs = qs[::-1]

        new_x = x
        for hm, q in zip(transform_list, qs):
            new_x = np.dot(hm(q), new_x)
        return (new_x)

    def loc_joints(self, qs):
        joint_abs_locations = []
        for idx, rel_loc in enumerate(self.joint_relative_locations):
            tmp_loc = self.forward_kinematics(qs, rel_loc, reference_frame=idx)
            joint_abs_locations.append(tmp_loc)
        return (np.asarray(joint_abs_locations))

    def configure_robots(self, qs):
        self.joint_abs_locations = self.loc_joints(qs)


# Q value function
def get_q_func(units_list, common_activation_func="relu", regularization=regularizers.l2(0.01)):
    """
    create a neural network model based on the units_list
    :param units_list: list of integers that specify the number of units in each layer
        The list has to contain at least 3 items.
    :return: a feedforward keras model object
    """
    input_dimension = units_list[0]
    model = Sequential()
    model.add(Dense(units_list[1], kernel_regularizer=regularization, input_dim=input_dimension))
    model.add(Activation(common_activation_func))
    for i, num_unit in enumerate(units_list[2:-1]):
        model.add(Dense(num_unit, kernel_regularizer=regularization))
        model.add(Activation(common_activation_func))
    model.add(Dense(units_list[-1], kernel_regularizer=regularization))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return(model)





