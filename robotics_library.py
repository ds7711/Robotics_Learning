# import libraries
import numpy as np
import itertools
import keras
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
    action_spaces = [[0]] * num_fixed_joints + \
                    [[1 * acceleration_resolution, 0, -1 * acceleration_resolution]] * (num_joints-num_fixed_joints) + \
                    [[0, 1]]
    action_combinations = np.asarray(list(itertools.product(*action_spaces)))  # enumerate all possible actions available
    ext_action_cmbs = np.hstack((np.zeros((len(action_combinations), state_dimension)),
                                 action_combinations))  # the first 12 columns are state, used for faster computations
    action_indexes = np.arange(len(action_combinations))

class Env2D(object):
    num_joints = 6
    num_fixed_joints = 4
    acceleration_resolution = 2.0 * np.pi / 360.0 / 5  # smallest acceleration one could apply at one time step
    state_dimension = num_joints * 2
    hoop_position = np.asarray([5, 3, 4]) # hoop position
    hoop_size = 1
    action_spaces = [[0]] * num_fixed_joints + \
                    [[1 * acceleration_resolution, 0, -1 * acceleration_resolution]] * (num_joints-num_fixed_joints) + \
                    [[0, 1]]
    action_combinations = np.asarray(list(itertools.product(*action_spaces)))  # enumerate all possible actions available
    ext_action_cmbs = np.hstack((np.zeros((len(action_combinations), state_dimension)),
                                 action_combinations))  # the first 12 columns are state, used for faster computations

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


env_dict = {
    "2d": Env2D,
    "3d": Env3D,
    "shaping": shaping
}



class Robotic_Manipulator_Naive(object):

    def __init__(self, link_lengthes, initial_angles, intial_angular_velocities, max_time=1000):
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
        self.time = 0
        self.max_time = max_time

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
        self.angular_velocities += action[:-1]
        self.release = action[-1]
        if self.time > self.max_time: # if too many time steps were executed, release the ball to stop training
            self.release = 1
            print("Maximum update step reached for this robot manipulator! Ball was released!!!")

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

def get_dist(pt1, pt2):
    return np.sqrt( (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2 );

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

    def __init__(self, q_obj, env_obj):

        self.action_combinations = env_obj.action_combinations
        self.ext_action_cmbs = env_obj.ext_action_cmbs
        self.action_indexes = np.arange(len(env_obj.action_combinations))
        self.q_obj = q_obj
        self.env_obj = env_obj

    def softmax_policy(self, state):
        """
        choose action based on the probability from softmax of the q value
        :param state:
        :param exploring_factor: balance exploration and exploitation
        :return:
        """
        self.ext_action_cmbs[:, :self.env_obj.state_dimension] = state
        est_q_values = self.q_obj.predict(self.env_obj.ext_action_cmbs)
        exponential_values = np.exp(est_q_values)
        probs = exponential_values / np.sum(exponential_values)
        action = self.env_obj.action_combinations[np.random.choice(self.action_indexes, p=np.squeeze(probs))]
        return(action)

    def epsilon_greedy_policy(state, q_obj, epsilon=0.8):
        """
        return the selected action based on the epsilon greedy
        :param state:
        :param q_obj:
        :param epsilon:
        :param action_combinations:
        :return:
        """
        pass


def generate_state_trajectory(robotic_arm, q_obj, reward_function, alpha, env_obj, discounting=0.9):
    """
    generate trajectories and training data from one trial
    :param robotic_arm:
    :param q_obj:
    :param alpha:
    :param discounting:
    :return:
    """
    # reward_threshold
    reward_threshold = np.exp(-alpha * env_obj.hoop_size) # convert the hoop size to reward threshold to test whether the ball is in

    # 1st step: initialize the policy object basedon the current q function
    policy_obj = Policy_Object(q_obj, env_obj)
    X = []
    Y = []

    # get the robot's initial state
    state = robotic_arm.state
    # select the 1st action based on the policy object
    action = policy_obj.softmax_policy(state[:-1])

    # initialize the state and action list
    state_list = [state]
    action_list = [action]

    # reward list
    while not robotic_arm.release: # if the ball was not released in the last time step, execute the following
        # get the first x: state-action pair
        tmp_x = np.concatenate((state[:-1], action))
        X.append(tmp_x)

        # update the robot based on the previous selected action
        robotic_arm.update_rm(action)
        # store the robot's next state
        next_state = robotic_arm.state
        # calculate the reward obtained from the next state
        reward = reward_function(robotic_arm, alpha, env_obj)
        if reward > 0:
            print(reward)
        # obtain the action after next state
        next_action = policy_obj.softmax_policy(next_state[:-1])
        next_q = q_obj.predict(np.concatenate((next_state[:-1], next_action))[np.newaxis, :])[0, 0]
        tmp_y = reward + discounting * next_q
        Y.append(tmp_y)

        state = next_state
        action = next_action
        state_list.append(state)
        action_list.append(action)
    score = float(reward > reward_threshold)
    reward = reward_function(robotic_arm, alpha, env_obj)
    return(np.asarray(X), np.asarray(Y), reward, score)


class Ball(object):

    def __init__(self, pos, vel, env_obj):
        self.pos = pos
        self.vel = vel
        self.min_dist_to_hoop = float("Inf")
        self.hoop_position = env_obj.hoop_position
    def update(self):
        t = 0.02 # set the time interval between two updates
        g = -10 # gravity acceleration
        ground = 0.01 # the ground
        while (self.pos[2] >= ground):
            # we assume the only acceleration caused by gravity is along z axis
            #so only z_dot changes
            self.vel[2] += g * t
            self.pos[0] += self.vel[0] * t
            self.pos[1] += self.vel[1] * t
            self.pos[2] += self.vel[2] * t
            temp_dist = get_dist(self.pos, self.hoop_position)
            if (temp_dist < self.min_dist_to_hoop):
                self.min_dist_to_hoop = temp_dist


def reward_function(robot_obj, alpha, env_obj):
    """
    reward function of the task
    :param robot_obj:
    :param alpha:
    :return:
    """
    if robot_obj.release == False: # if ball was in hold, 0 reward
        return(0)
    elif robot_obj.release == True: # if ball was released, calculate the ball's distance to the hoop when it crosses the plane of the hoop
        pos = robot_obj.loc_joints()[-1] # get ee position
        vel = robot_obj.cal_ee_speed() # get ee velocity
        tmp_ball = Ball(pos[:3], vel[:3], env_obj)
        tmp_ball.update()
        reward = np.exp(-alpha * tmp_ball.min_dist_to_hoop)
        return(reward)
    else:
        print("Errors in reward function!!!")
        return(None)



# Neural-fitted Q-value algorithm
def neural_fitted_q_algorithm(ini_robot_obj, q_obj, env_obj, reward_func=reward_function,
                              num_iterations=3, minimum_samples=200, minimum_trj=3, verbose=0,
                              model_name="bk_model.h5"):
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
    link_lengthes = ini_robot_obj.link_lengthes
    initial_angles = ini_robot_obj.joint_angles
    initial_angular_velocities = ini_robot_obj.angular_velocities

    reward_list = []
    score_list = []
    for iii in range(num_iterations):

        # 1st step: create the training data
        X_list = []
        Y_list = []
        num_trajectories = 0
        while len(X_list) < minimum_samples or num_trajectories < minimum_trj:
            robot_obj = Robotic_Manipulator_Naive(link_lengthes, initial_angles, initial_angular_velocities)
            X, Y, reward, score = generate_state_trajectory(robot_obj, q_obj, reward_func, alpha=1, env_obj=env_obj)
            X_list.extend(X)
            Y_list.extend(Y)
            reward_list.append(reward)
            score_list.append(score)

            num_trajectories += 1
        X_list = np.vstack(X_list)
        Y_list = np.asarray(Y_list)

        # 2nd step: train the q_object
        q_obj.fit(X_list, Y_list, batch_size=minimum_samples, epochs=len(X_list) / minimum_samples * minimum_trj, verbose=verbose)
    q_obj.save(model_name)
    return(q_obj, reward_list, score_list)












