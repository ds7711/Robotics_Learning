# import libraries
import numpy as np
import itertools
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers

import tensorflow as tf


# some constants
num_joints = 6
acceleration_resolution = 2 * np.pi / 360.0 # smallest acceleration one could apply at one time step


# for visualization purpose only, to delete
acceleration_resolution = 1e-3
# for visualization purpose only, to delete


# the first 6 lists specifying the action space of the joints, the last one specifies the action for end-effector, 0=hold, 1=release
# in the first test, the first 4 joints should only have 0 actions
action_spaces = [[1 * acceleration_resolution, 0, -1 * acceleration_resolution]] * num_joints + [[0, 1]]
action_combinations = np.asarray(list(itertools.product(*action_spaces))) # enumerate all possible actions available



class Robotic_Manipulator_Naive(object):

    def __init__(self, link_lengthes, initial_angles,
                 intial_angular_velocities=np.zeros(num_joints) # initial angular velocities of the robots
                 ):
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
        self.joint_angles = initial_angles
        self.angular_velocities = intial_angular_velocities
        self.release = False
        self.num_joints = len(initial_angles)
        self.rotation_limit = None # to add the maximum achieve joint angles

        # a list of homogeneous transformation matrix functions
        def r10(q, idx=0):
            initial_q = self.joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), -np.sin(q), 0, 0],
                             [np.sin(q), np.cos(q), 0, 0],
                             [0, 0, 1, l],
                             [0, 0, 0, 1]])
            return (hm)

        def r21(q, idx=1):
            initial_q = self.joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[1, 0, 0, 0],
                             [0, np.cos(q), -np.sin(q), 0],
                             [0, np.sin(q), np.cos(q), l],
                             [0, 0, 0, 1]])
            return (hm)

        def r32(q, idx=2):
            initial_q = self.joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[1, 0, 0, 0],
                             [0, np.cos(q), -np.sin(q), l],
                             [0, np.sin(q), np.cos(q), 0],
                             [0, 0, 0, 1]])
            return (hm)

        def r43(q, idx=3):
            initial_q = self.joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), -np.sin(q), 0, 0],
                             [np.sin(q), np.cos(q), 0, l],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
            return (hm)

        def r54(q, idx=4):
            initial_q = self.joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), 0, -np.sin(q), 0],
                             [0, 1, 0, l],
                             [np.sin(q), 0, np.cos(q), 0],
                             [0, 0, 0, 1]])
            return (hm)

        def r65(q, idx=5):
            initial_q = self.joint_angles[idx]
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
        """
        convert the relative loctions in the joint frame into the world frame
        :param qs: joint angles
        :param x: location in the joint frame
        :param reference_frame: frame in which the x is specified
        :return:
        """
        transform_list = self.ht_list[:reference_frame]
        transform_list = transform_list[::-1]

        qs = qs[:reference_frame]
        qs = qs[::-1]

        new_x = x
        for hm, q in zip(transform_list, qs):
            new_x = np.dot(hm(q), new_x)
        return (new_x)

    def loc_joints(self, qs=None):
        """
        if qs is not specified, calculate the absolute positions of the joints under the current configurations
        :param qs:
        :return:
        """
        if qs is None: # if joint angles were not specified, use the current joint angles
            qs = self.joint_angles
        joint_abs_locations = []
        for idx, rel_loc in enumerate(self.joint_relative_locations):
            tmp_loc = self.forward_kinematics(qs, rel_loc, reference_frame=idx)
            joint_abs_locations.append(tmp_loc)
        self.joint_abs_locations = np.asarray(joint_abs_locations)
        return (self.joint_abs_locations)

    def configure_robots(self, qs):
        self.joint_abs_locations = self.loc_joints(qs)

    def _update_angular_velocities(self, action):
        self.angular_velocities += action[:num_joints]
        self.release = action[-1]

    def _update_joint_angles(self):
        self.joint_angles += self.angular_velocities

    def update_rm(self, action):
        """
        update the robot to the next time step
        update the robot's joint angles based on the velocity from previous time step and \
            angular velocity based on current action (specified as acceleration)
        :param action: acceleration at each joint and whether to release the ball
        :return:
        """
        self._update_joint_angles()
        self._update_angular_velocities(action)

    def _jacobian_matrix(self):
        """
        calculate the value of the Current Jacobian matrix values
        :return:
        """
        pass

    def cal_ee_speed(self):
        """
        calculate the speed of the end effector in the world frame
        :return:
        """
        pass



# Q value function object
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


# Neural-fitted Q-value algorithm
def q_algorithm(robot_ojb, q_obj, policy_obj, reward_func):
    pass
    """
    # 1st: move the ball based on the q_ojbect function and policy_object function and record every trajectories
    # each trajectory includes: [(s_t, a_t, R_t, s_t+1, a_t+1)] for t = 0 to H, different trajectories are indexed by i

    # 2nd: convert the trajectories into a format that can be used to train the q_object
    #  x_t^i = (s_t, a_t)
    #  y_t^i = R_t +  discounting * q_object(s_t+1, a_t+1)
    # do this for every time step in every trajectory and put them into big X and Y
    #   where, each row of X is (s_t, a_t)

    # 3rd: train q_object using the X and y as the training data
    #   use fit or train_on_batch method of the q_obj model
    #   update q_obj

    # Repeat the above steps until q_obj converge
    """



"""
Things to do:
    1. Implement the policy_object or function and reward function so that new trajectories can be generated
        a. Define the domain of the actions, how many discrete actions are available in each joint
        b. How to define the reward function
        c. How to determine the terminal state
        d. Implement the greedy policy function (more efficient policy function may be required)

    2. Implement the function that convert the trajectories to the training data

"""







