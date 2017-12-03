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
state_dimension = num_joints * 2

# for visualization purpose only, to delete
acceleration_resolution = 1e-3
# for visualization purpose only, to delete


# the first 6 lists specifying the action space of the joints, the last one specifies the action for end-effector, 0=hold, 1=release
# in the first test, the first 4 joints should only have 0 actions
action_spaces = [[1 * acceleration_resolution, 0, -1 * acceleration_resolution]] * num_joints + [[0, 1]]
action_combinations = np.asarray(list(itertools.product(*action_spaces))) # enumerate all possible actions available
ext_action_cmbs = np.hstack((np.zeros((len(action_combinations), state_dimension)), action_combinations)) # the first 12 columns are state, used for faster computations
action_indexes = np.arange(len(action_combinations))

# hoop_position
hoop_position = np.asarray([10, 10, 10])



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
        self.state = np.concatenate((self.joint_angles, self.angular_velocities, [self.release]))

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

    def _simple_forward_kinematics(self, qs, new_x, transform_list):
        """
        routine for calculating the absolute location
        :param new_x:
        :param transform_list:
        :param qs:
        :return:
        """
        for hm, q in zip(transform_list, qs):
            new_x = np.dot(hm(q), new_x)
        return (new_x)

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

        new_x = self._simple_forward_kinematics(qs, x, transform_list)
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
        self.state[:self.num_joints] = self.joint_angles
        self.state[self.num_joints:-1] = self.angular_velocities
        self.state[-1] = self.release


    def _jacobian_matrix(self, x, qs=None, reference_frame=6, delta=1e-10):
        """
        calculate the value of the Current Jacobian matrix values
        :return:
        """
        if qs is None:
            qs = self.joint_angles
        new_qs = qs[:reference_frame]
        new_qs = new_qs[::-1]

        transform_list = self.ht_list[:reference_frame]

        new_x = self._simple_forward_kinematics(new_qs, x, transform_list)

        jacobian = []
        delta_qs = np.zeros(len(transform_list))
        for i in range(len(transform_list)):
            delta_qs[i] = delta
            tmp_qs = new_qs + delta_qs
            tmp_x = self._simple_forward_kinematics(tmp_qs, x, transform_list)
            tmp_jacobian = (tmp_x - new_x) / delta
            jacobian.append(tmp_jacobian)
        return(np.asarray(jacobian))

    def cal_ee_speed(self):
        """
        calculate the speed of the end effector in the world frame
        :return:
        """
        x = self.joint_relative_locations[-1]
        jacobian = self._jacobian_matrix(x)
        v_dot = np.dot(jacobian.T, self.angular_velocities)
        return(v_dot)



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


class Policy_Object(object):

    def __init__(self, q_obj, action_combinations=action_combinations, ext_action_cmbs=ext_action_cmbs):

        self.action_combinations = action_combinations
        self.ext_action_cmbs = ext_action_cmbs
        self.action_indexes = np.arange(len(action_combinations))
        self.q_obj = q_obj

    def softmax_policy(self, state):
        self.action_combinations[:, state_dimension] = state
        est_q_values = self.q_obj.predict(ext_action_cmbs)
        exponential_values = np.exp(est_q_values)
        probs = exponential_values / np.sum(exponential_values)
        action = action_combinations[np.random.choice(action_indexes, p=probs)]
        return(action)

    def epsilon_greedy_policy(state, q_obj, epsilon=0.8, action_combinations=ext_action_cmbs):
        """
        return the selected action based on the epsilon greedy
        :param state:
        :param q_obj:
        :param epsilon:
        :param action_combinations:
        :return:
        """
        pass


def generate_state_trajectory(robotic_arm, q_obj, threshold, discounting=0.9):
    """
    generate trajectories and training data from one trial
    :param robotic_arm:
    :param q_obj:
    :param threshold:
    :param discounting:
    :return:
    """
    policy_obj = Policy_Object(q_obj)
    X = []
    Y = []

    state = robotic_arm.state
    action = policy_obj.softmax_policy(state)
    state_list = [state]
    action_list = [action]
    while not state.release:
        tmp_x = np.concatenate((state[:-1], action))
        X.append(tmp_x)

        robotic_arm.update_rm(action)
        next_state = robotic_arm.state

        reward = reward_function(next_state, hoop_position, threshold)

        next_action = policy_obj.softmax_policy(next_state)
        tmp_y = reward + discounting * q_obj.predict(next_state[:-1])
        Y.append(tmp_y)

        state = next_state
        action = next_action
        state_list.append(state)
        action_list.append(action)

    return(np.asarray(X), np.asarray(Y))


def ball_distance(state, hoop_position):
    pass


def reward_function(state, hoop_position, threshold):
    if state[-1] == 0: # if ball was in hold, 0 reward
        return(0)
    elif state[-1] == 1: # if ball was released, calculate the ball's distance to the hoop when it crosses the plane of the hoop
        reward = float(ball_distance(state, hoop_position) < threshold)
        return(reward)
    else:
        print("Errors in reward function!!!")
        return(None)



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





# def epsilon_greedy_policy(state, q_obj, epsilon=0.8, action_combinations=ext_action_cmbs):
#     """
#     return the selected action based on the epsilon greedy
#     :param state:
#     :param q_obj:
#     :param epsilon:
#     :param action_combinations:
#     :return:
#     """
#
#     pass

# def softmax_policy(state, q_obj,  action_combinations=action_combinations, ext_action_cmbs=ext_action_cmbs,
#                    action_indexes=action_indexes):
#     """
#     return the selected action based on the softmax probability
#     :param state:
#     :param q_obj:
#     :param action_combinations:
#     :return:
#     """
#     action_combinations[:, state_dimension] = state
#     est_q_values = q_obj.predict(ext_action_cmbs)
#     exponential_values = np.exp(est_q_values)
#     probs = exponential_values / np.sum(exponential_values)
#     action = action_combinations[np.random.choice(action_indexes, p=probs)]
#     return(action)







