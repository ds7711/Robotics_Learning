# import libraries
import numpy as np
import itertools
import keras
import copy
import pdb
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers

import tensorflow as tf


# environment


class Env3D(object):
    num_joints = 6
    num_fixed_joints = 0
    acceleration_resolution = 2.0 * np.pi / 360.0 / 5  # smallest acceleration one could apply at one time step
    state_dimension = num_joints * 2
    hoop_position = np.asarray([10, 3, 10]) # hoop position
    hoop_size = 1
    score_threshold = 0.5
    action_spaces = [[0]] * num_fixed_joints + \
                    [[1 * acceleration_resolution, 0, -1 * acceleration_resolution]] * (num_joints-num_fixed_joints) + \
                    [[0, 1]]
    action_combinations = np.asarray(list(itertools.product(*action_spaces)))  # enumerate all possible actions available
    ext_action_cmbs = np.hstack((np.zeros((len(action_combinations), state_dimension)),
                                 action_combinations))  # the first 12 columns are state, used for faster computations
    action_indexes = np.arange(len(action_combinations))

class shaping(object):
    num_joints = 6
    num_fixed_joints = 4
    acceleration_resolution = 2.0 * np.pi / 360.0 / 5  # smallest acceleration one could apply at one time step
    state_dimension = num_joints * 2
    hoop_position = np.asarray([5, 3, 4]) # hoop position
    hoop_size = 1
    action_spaces = [[0]] * num_fixed_joints + \
                    [[1 * acceleration_resolution, 0, -1 * acceleration_resolution]] * (num_joints-num_fixed_joints) + \
                    [[0]]
    action_combinations = np.asarray(list(itertools.product(*action_spaces)))  # enumerate all possible actions available
    ext_action_cmbs = np.hstack((np.zeros((len(action_combinations), state_dimension)),
                                 action_combinations))  # the first 12 columns are state, used for faster computations

class Env2D(object):

    def __init__(self, hoop_size=4, policy_greedy=1):
        # configurations for the robotic arm
        self.num_joints = 6
        self.num_fixed_joints = 4
        self.acceleration_resolution = 2.0 * np.pi / 360.0 / 5  # smallest acceleration one could apply at one time step
        self.state_dimension = self.num_joints * 2

        self.link_lengthes = [1, 3, 1, 1, 1, 1, 2]
        self.initial_angles = [0, 0, 0, 0, -np.pi / 4, 0]
        self.initial_angular_velocities = np.zeros(self.num_joints)

        # configuration for the hoop
        self.hoop_position = np.asarray([5, 3, 2]) # hoop position
        self.dist_threshold = 0.85 # fixed to check the performance

        # specify the action spaces
        self.action_spaces = [[0]] * self.num_fixed_joints + \
                             [list(np.asarray([25, 10, 3, 1, 0, -1, -3, -10, -25]) * self.acceleration_resolution)] * \
                             (self.num_joints-self.num_fixed_joints) + \
                             [[0, 1]]
        # self.action_spaces = [[0]] * self.num_fixed_joints + \
        #                      [list(np.arange(-10, 11, 5) * self.acceleration_resolution)] * \
        #                      (self.num_joints-self.num_fixed_joints) + \
        #                      [[0, 1]]
        # pdb.set_trace()
        self.action_combinations = np.asarray(list(itertools.product(*self.action_spaces)))  # enumerate all possible actions available
        self.ext_action_cmbs = np.hstack((np.zeros((len(self.action_combinations), self.state_dimension)),
                                     self.action_combinations))  # the first 12 columns are state, used for faster computations

        # parameters used for training
        self.hoop_size = hoop_size  # for training, decrease as training goes
        self.policy_greedy = policy_greedy # change the steepness of the softmax function of the policy object
        self.epsilon = 1.0 / len(self.action_combinations) # parameter for epsilon-greedy policy function
        self.epsilon = 0.75
        self.epsilon_increase = 0.25 # epsilon increase proportion when perfance reaches a threshold
        self.alpha = 1 # controls the relationship between reward and distance to the hoop
        self.max_reward = 100
        self.noise_level = 0.1




class Robotic_Manipulator_Naive(object):

    # to add time resolution into the manipulator

    def __init__(self, link_lengthes, initial_angles, intial_angular_velocities, max_time=1000):
        """
        create the robotic manipulator object
        :param link_lengthes: 
        :param initial_angles: 
        :param intial_angular_velocities: 
        :param max_time: 
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
        self.initial_joint_angles = np.asarray(initial_angles, dtype=np.double)
        self.joint_angles = np.asarray(initial_angles, dtype=np.double)
        self.angular_velocities = np.asarray(intial_angular_velocities, dtype=np.double)
        self.release = False
        self.num_joints = len(initial_angles)
        self.rotation_limit = None # to add the maximum achieve joint angles
        self.state = np.concatenate((self.joint_angles, self.angular_velocities, [self.release]))
        self.time = 0
        self.max_time = max_time
        self.joint_angle_limit = np.pi / 2 * 10
        self.joint_vel_limit = 2 * np.pi / 360.0 * 20.0 * 20

        # a list of homogeneous transformation matrix functions
        def r10(q, idx=0):
            initial_q = self.initial_joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), -np.sin(q), 0, 0],
                             [np.sin(q), np.cos(q), 0, 0],
                             [0, 0, 1, l],
                             [0, 0, 0, 1]])
            return (hm)

        def r21(q, idx=1):
            initial_q = self.initial_joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[1, 0, 0, 0],
                             [0, np.cos(q), -np.sin(q), 0],
                             [0, np.sin(q), np.cos(q), l],
                             [0, 0, 0, 1]])
            return (hm)

        def r32(q, idx=2):
            initial_q = self.initial_joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[1, 0, 0, 0],
                             [0, np.cos(q), -np.sin(q), l],
                             [0, np.sin(q), np.cos(q), 0],
                             [0, 0, 0, 1]])
            return (hm)

        def r43(q, idx=3):
            initial_q = self.initial_joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), -np.sin(q), 0, 0],
                             [np.sin(q), np.cos(q), 0, l],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
            return (hm)

        def r54(q, idx=4):
            initial_q = self.initial_joint_angles[idx]
            l = self.link_lengthes[idx]
            q += initial_q
            hm = np.asarray([[np.cos(q), 0, -np.sin(q), 0],
                             [0, 1, 0, l],
                             [np.sin(q), 0, np.cos(q), 0],
                             [0, 0, 0, 1]])
            return (hm)

        def r65(q, idx=5):
            initial_q = self.initial_joint_angles[idx]
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
            tmp_x = np.dot(hm(q), new_x)
            new_x = tmp_x
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
        """
        configure the robot to specified rotation angles
        :param qs: specify the angles of each joints
        :return: 
        """
        self.joint_angles = qs
        self.joint_abs_locations = self.loc_joints(self.joint_angles)

    def _update_angular_velocities(self, action):
        """
        update the angular velocities and hold/release state
        :param action: 
        :return: 
        """
        self.angular_velocities += action[:-1]
        self.release = action[-1]
        if np.any(np.abs(self.angular_velocities) > self.joint_vel_limit):
            self.release = 1

    def _update_joint_angles(self):
        """
        update the joint angles
        :return: 
        """
        self.joint_angles += self.angular_velocities
        if np.any(np.abs(self.joint_angles) > self.joint_angle_limit):
            self.release = 1

    def update_rm(self, action):
        """
        update the robot to the next time step
        update the robot's joint angles based on the velocity from previous time step and \
            angular velocity based on current action (specified as acceleration)
        :param action: acceleration at each joint and whether to release the ball
        :return:
        """
        self._update_joint_angles()
        # pdb.set_trace()
        self._update_angular_velocities(action)
        self.state[:self.num_joints] = self.joint_angles
        self.state[self.num_joints:-1] = self.angular_velocities
        self.state[-1] = self.release
        self.time += 1


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
        transform_list = transform_list[::-1]

        new_x = self._simple_forward_kinematics(new_qs, x, transform_list)

        jacobian = np.zeros((4, self.num_joints))
        for i in range(self.num_joints):
            delta_qs = np.zeros(len(transform_list))
            delta_qs[i] = delta
            tmp_qs = new_qs + delta_qs
            tmp_x = self._simple_forward_kinematics(tmp_qs, x, transform_list)
            tmp_jacobian = (tmp_x - new_x) / delta
            jacobian[:, self.num_joints-i-1] = tmp_jacobian
            # jacobian.append(tmp_jacobian)
        return(jacobian)

    def cal_ee_speed(self):
        """
        calculate the speed of the end effector in the world frame
        :return:
        """
        # pdb.set_trace()
        x = self.joint_relative_locations[-1]
        jacobian = self._jacobian_matrix(x)
        v_dot = np.dot(jacobian, self.angular_velocities)
        return(v_dot)

def get_dist(pt1, pt2):
    """
    calculate distance between two points in 3D space
    :param pt1:
    :param pt2:
    :return:
    """
    return np.sqrt( (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2 )

class Ball(object):

    def __init__(self, pos, vel, env_obj):
        """
        initialize the ball object
        :param pos: 
        :param vel: 
        :param env_obj: 
        """
        self.pos = pos
        self.vel = vel
        self.min_dist_to_hoop = float("Inf")
        self.hoop_position = env_obj.hoop_position
    def update(self):
        t = 0.01 # set the time interval between two updates
        g = -10 # gravity acceleration
        ground = 0.01 # the ground
        ball_trajectory = []
        while (self.pos[2] > ground):
            # we assume the only acceleration caused by gravity is along z axis
            #so only z_dot changes
            ball_trajectory.append(np.copy(self.pos)) # store the trajectory of the ball

            self.pos[0] += self.vel[0] * t
            self.pos[1] += self.vel[1] * t
            self.pos[2] += self.vel[2] * t
            self.vel[2] += g * t

            temp_dist = get_dist(self.pos, self.hoop_position)
            if (temp_dist < self.min_dist_to_hoop):
                self.min_dist_to_hoop = temp_dist
        return(np.asarray(ball_trajectory))


def reward_function(robot_obj, env_obj):
    """
    reward function of the task
    :param robot_obj:
    :param alpha:
    :return:
    """
    reward = 0

    if robot_obj.release == False: # if ball was in hold, 0 reward
        return(reward)

    elif robot_obj.release == True: # if ball was released, calculate the ball's distance to the hoop when it crosses the plane of the hoop
        pos = robot_obj.loc_joints()[-1] # get ee position
        vel = robot_obj.cal_ee_speed() # get ee velocity
        tmp_ball = Ball(pos[:3], vel[:3], env_obj)
        tmp_ball.update()

        # if the ball is close to the hoop (distance closer than hoop size), return a reward
        if tmp_ball.min_dist_to_hoop < env_obj.hoop_size:
            reward = np.exp(- (tmp_ball.min_dist_to_hoop * env_obj.alpha)) * env_obj.max_reward
            # reward = 1.0 / (1 + tmp_ball.min_dist_to_hoop)
            if tmp_ball.min_dist_to_hoop < env_obj.dist_threshold:
                print("Successful throw!!!", tmp_ball.min_dist_to_hoop)
                # pdb.set_trace()
        # pdb.set_trace()
        return(reward)
    else:
        print("Errors in reward function!!!")
        return(None)


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

    def __init__(self, q_obj, env_obj, policy_type="softmax"):

        self.action_combinations = env_obj.action_combinations
        self.ext_action_cmbs = env_obj.ext_action_cmbs
        self.action_indexes = np.arange(len(env_obj.action_combinations))
        self.q_obj = q_obj
        self.env_obj = copy.deepcopy(env_obj)
        self.greedy = self.env_obj.policy_greedy
        self.policy_type = policy_type

    def softmax_policy(self, state):
        """
        choose action based on the probability from softmax of the q value
        :param state:
        :param exploring_factor: balance exploration and exploitation
        :return:
        """
        self.ext_action_cmbs[:, :self.env_obj.state_dimension] = state
        est_q_values = self.q_obj.predict(self.env_obj.ext_action_cmbs)
        exponential_values = np.exp(est_q_values * self.greedy)
        probs = exponential_values / np.sum(exponential_values)
        action = self.env_obj.action_combinations[np.random.choice(self.action_indexes, p=np.squeeze(probs))]
        return(action)

    def epsilon_greedy_policy(self, state):
        """
        return the selected action based on the epsilon greedy
        :param state:
        :param q_obj:
        :param epsilon:
        :param action_combinations:
        :return:
        """
        self.ext_action_cmbs[:, :self.env_obj.state_dimension] = state
        est_q_values = self.q_obj.predict(self.env_obj.ext_action_cmbs)
        if np.random.rand() < self.env_obj.epsilon:
            act_idx = np.argmax(est_q_values)
            return(self.env_obj.action_combinations[act_idx])
        else:
            act_idx = np.random.choice(self.action_indexes)
            return (self.env_obj.action_combinations[act_idx])

    def select_action(self, state):
        if self.policy_type == "softmax":
            action = self.softmax_policy(state)
        elif self.policy_type == "epsilon_greedy":
            action = self.epsilon_greedy_policy(state)
        else:
            print("Errors in policy type of the reward function!!!")
        action = noise_in_action(action, env_obj=self.env_obj)
        return(action)


def test_q_function(q_obj, env_obj_old, num_test=100, policy_type="epsilon_greedy"):
    """
    test the performance of the policy based on the learned q function
    :param robotic_arm: 
    :param q_obj: 
    :param reward_function: 
    :param alpha: 
    :param env_obj: 
    :return: 
    """
    env_obj = copy.deepcopy(env_obj_old)
    env_obj.epsilon = 0.95
    env_obj.hoop_size = 3
    env_obj.alpha = 2.5

    score_threshold = np.exp(-env_obj.alpha * env_obj.dist_threshold) * env_obj.max_reward # convert the hoop size to reward threshold to test whether the ball is in
    # 1st step: initialize the policy object basedon the current q function

    policy_obj = Policy_Object(q_obj, env_obj, policy_type=policy_type)

    ee_trajectory_list = []
    ee_final_pos_list = []
    ee_speed_list = []
    sum_reward_list = []
    score_list = []
    for _ in range(num_test):
        link_lengthes = np.copy(env_obj.link_lengthes)
        initial_angles = np.copy(env_obj.initial_angles)
        initial_angular_velocities = np.copy(env_obj.initial_angular_velocities)
        robotic_arm = Robotic_Manipulator_Naive(link_lengthes, initial_angles,
                                                initial_angular_velocities)
        # get the robot's initial state
        state = robotic_arm.state
        # select the 1st action based on the policy object
        # pdb.set_trace()
        action = policy_obj.select_action(state[:-1])

        # initialize the state and action list
        state_list = []
        action_list = []
        reward_list = []
        ee_pos_list = [robotic_arm.loc_joints()[-1]]

        # reward list
        while not robotic_arm.release: # if the ball was not released in the last time step, execute the following
            # get the first x: state-action pair
            state_list.append(np.copy(state))
            action_list.append(np.copy(action))

            # update the robot based on the previous selected action
            robotic_arm.update_rm(action)
            # store the robot's next state
            next_state = robotic_arm.state
            ee_pos_list.append(np.copy(robotic_arm.loc_joints()[-1]))
            if robotic_arm.release == 1:
                ee_speed = robotic_arm.cal_ee_speed()
            # calculate the reward obtained from the next state
            reward = reward_function(robotic_arm, env_obj)
            reward_list.append(reward)
            # obtain the action after next state
            next_action = policy_obj.select_action(next_state[:-1])
            # next_q = q_obj.predict(np.concatenate((next_state[:-1], next_action))[np.newaxis, :])[0, 0]
            # q_val_list.append(next_q)

            state = next_state
            action = next_action

        state_list.append(state)
        action_list.append(action)
        score = float(reward_list[-1] > score_threshold)
        reward_sum = np.sum(reward_list)

        # add to the test list
        ee_trajectory_list.append(ee_pos_list)
        ee_final_pos_list.append(ee_pos_list[-1])
        ee_speed_list.append(ee_speed)
        sum_reward_list.append(reward_sum)
        score_list.append(score)

    return(ee_trajectory_list, ee_final_pos_list, ee_speed_list,
           sum_reward_list, score_list)

def noise_in_action(action, env_obj):
    """
    to add
    :param action: 
    :return: 
    """
    new_action = np.copy(action)
    for iii in range(env_obj.num_fixed_joints, env_obj.num_joints, 1):
        tmp_noise = np.random.randn() * action[iii] * env_obj.noise_level
        new_action[iii] += tmp_noise
    return(new_action)

def generate_state_trajectory(q_obj, reward_func, env_obj,
                              policy_type="epsilon_greedy"):
    """
    generate trajectories and training data from one trial
    :param robotic_arm:
    :param q_obj:
    :param alpha:
    :param discounting:
    :return:
    """
    link_lengthes = np.copy(env_obj.link_lengthes)
    initial_angles = np.copy(env_obj.initial_angles)
    initial_angular_velocities = np.copy(env_obj.initial_angular_velocities)
    robotic_arm = Robotic_Manipulator_Naive(link_lengthes, initial_angles, initial_angular_velocities)

    # reward_threshold
    alpha = env_obj.alpha
    score_threshold = env_obj.max_reward * np.exp(-alpha * env_obj.dist_threshold) # convert the hoop size to reward threshold to test whether the ball is in

    # 1st step: initialize the policy object basedon the current q function
    policy_obj = Policy_Object(q_obj, env_obj, policy_type=policy_type)

    # get the robot's initial state
    state = robotic_arm.state
    # select the 1st action based on the policy object
    action = policy_obj.select_action(state[:-1])

    # initialize the state and action list
    state_list = []
    action_list = []
    reward_list = []
    # q_val_list = []

    # reward list
    while not robotic_arm.release: # if the ball was not released in the last time step, execute the following
        # get the first x: state-action pair
        state_list.append(np.copy(state))
        action_list.append(np.copy(action))

        # update the robot based on the previous selected action
        robotic_arm.update_rm(action)
        # store the robot's next state
        next_state = robotic_arm.state
        # calculate the reward obtained from the next state
        reward = reward_func(robotic_arm, env_obj)
        reward_list.append(reward)
        # obtain the action after next state
        next_action = policy_obj.select_action(next_state[:-1])
        # next_q = q_obj.predict(np.concatenate((next_state[:-1], next_action))[np.newaxis, :])[0, 0]
        # q_val_list.append(next_q)

        state = next_state
        action = next_action

    state_list.append(state)
    action_list.append(action)
    score = float(reward_list[-1] > score_threshold)
    if score > 0.1:
        print("Score added!")
    reward = reward_func(robotic_arm, env_obj)
    # pdb.set_trace()
    # print "hello"
    return(np.asarray(state_list), np.asarray(action_list), np.asarray(reward_list), reward, score)


def trajectory2data(state_list, action_list, reward_list, q_object, discounting_factor=0.95):
    # print(state_list.shape, action_list.shape)
    all_states = np.hstack((state_list[:, :-1], action_list))
    X = all_states[:-1, :]
    q_values = np.squeeze(q_object.predict(all_states[1:, :]))
    q_values[-1] = 0
    Y = reward_list + discounting_factor * q_values
    # print X.shape, Y.shape
    return(X, Y)


# Neural-fitted Q-value algorithm
def neural_fitted_q_algorithm(q_obj, env_obj, data_pool, reward_func=reward_function,
                              num_iterations=5, minimum_samples=200, minimum_trj=5, verbose=0,
                              model_name="bk_model.h5", policy_type="epsilon_greedy"):
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
    # get the initial parameters of the robot

    for iii in range(num_iterations):

        # 1st step: create the training data
        X_list = []
        Y_list = []
        score_list = []
        final_reward_list = []
        num_trajectories = 0
        while len(X_list) < minimum_samples or num_trajectories < minimum_trj:
            state_list, action_list, reward_list, reward, score = generate_state_trajectory(q_obj,
                                                                                            reward_func,
                                                                                            env_obj=copy.deepcopy(env_obj),
                                                                                            policy_type=policy_type)
            if len(state_list) > 10 and reward > 50:
                pdb.set_trace()
            if len(state_list) > 2: # at least two actions have to be made to be considered as a valid trajectory
                X, Y = trajectory2data(state_list, action_list, reward_list, q_obj)
                # print X.shape, Y.shape
                # # print X, Y
                # print(len(X_list), len(Y_list))
                X_list.append(np.copy(X))
                Y_list.append(np.copy(Y))
                # X, Y, reward, score = generate_state_trajectory(robot_obj, q_obj, reward_func, alpha=1, env_obj=env_obj)
                score_list.append(score)
                final_reward_list.append(reward)
                if len(reward_list) > 3 and reward > -1:
                    print(reward_list)
                    # print(env_obj.hoop_size)

                num_trajectories += 1
                data_pool.add(state_list, action_list, reward_list, q_obj)
        X_train = np.vstack(X_list + data_pool.X_list)
        Y_train = np.concatenate(Y_list + data_pool.Y_list)

        # 2nd step: train the q_object
        q_obj.fit(X_train, Y_train, batch_size=len(X_train) / minimum_trj,
                  epochs=minimum_trj * 5, verbose=verbose)
        q_obj.save(model_name)
    return(q_obj, final_reward_list, score_list)


class DataPool(object):
    """
    create a container to store the positive trajectories
    """

    def __init__(self, q_obj, max_trajectories=100):
        self.X_list = []
        self.Y_list = []
        self.rewards = []
        self.minimum_rewards = 1e-10
        self.max_trajectories = max_trajectories
        self.num_trajectories = 0
        self.q_obj = q_obj


    def add(self, state_list, action_list, reward_list, q_obj, discounting_factor=0.95):
        if reward_list[-1] > self.minimum_rewards:
            # pdb.set_trace()
            X, Y = trajectory2data(state_list, action_list, reward_list, q_obj, discounting_factor=discounting_factor)
            if self.num_trajectories < self.max_trajectories:
                self.X_list.append(np.copy(X))
                self.Y_list.append(np.copy(Y))
                self.rewards.append(reward_list[-1])
                self.num_trajectories += 1
            else:
                # pdb.set_trace()
                min_idx = np.argmin(self.rewards)
                self.X_list[min_idx] = X
                self.Y_list[min_idx] = Y
                self.rewards[min_idx] = reward_list[-1]



def shaping_training(q_obj, env_obj, data_pool, shaping_factor=1.4, reward_func=reward_function,
                     num_iterations=3, minimum_samples=200, minimum_trj=10, verbose=0,
                     model_name="bk_model.h5", policy_type="softmax"):

    iii = 1
    _, _, _, reward, score = test_q_function(q_obj, env_obj)
    max_score = 0
    while np.mean(score) < 0.1:
        new_q_obj, reward_list, score_list = neural_fitted_q_algorithm(q_obj, env_obj, data_pool,
                                                                       reward_func=reward_func,
                                                                       num_iterations=num_iterations, minimum_samples=minimum_samples,
                                                                       minimum_trj=minimum_trj, verbose=verbose,
                                                                       model_name=model_name, policy_type=policy_type)
        _, _, _, reward, score = test_q_function(new_q_obj, env_obj)
        if np.mean(reward) > max_score:
            max_score = np.mean(score)
            env_obj.hoop_size = env_obj.hoop_size / shaping_factor
            # env_obj.policy_greedy += shaping_factor
            # env_obj.epsilon += env_obj.epsilon_increase * (1 - env_obj.epsilon)
            # env_obj.epsilon = env_obj.epsilon ** 3

        print("-------------------------------------------------------------------------")
        print(iii, np.mean(reward), np.mean(score), env_obj.hoop_size, env_obj.epsilon, env_obj.policy_greedy)
        print("-------------------------------------------------------------------------")

        q_obj = new_q_obj # update to new q_object
        iii += 1
        if iii % 2 == 0:
            pdb.set_trace()
            pass

    print("Training successfully completed!")
    return(q_obj, reward_list, score_list)














