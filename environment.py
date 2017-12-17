import numpy as np
import itertools
import kinematics_library as knl


class Env2D(object):

    def __init__(self):
        # configurations for the robotic arm
        self.num_joints = 6
        self.num_fixed_joints = 4
        self.state_dimension = self.num_joints * 2

        self.link_lengthes = np.asarray([1, 3, 1, 1, 1, 1, 1.0])
        self.initial_angles = np.asarray([0, 0, 0, 0, -np.pi / 2, 0])
        self.initial_angular_velocities = np.zeros(self.num_joints)

        # initial configuration of the robot
        self.robot_time_step = 1.0e-3
        self.ini_ra = knl.RobotArm(self.link_lengthes, self.initial_angles, self.initial_angular_velocities,
                                   time_step=self.robot_time_step)
        self.ini_ee_pos = self.ini_ra.loc_joints()[-1][:-1]

        # configuration for the hoop
        self.hoop_position = np.asarray([4.72, 3, 0.5], dtype=np.double)  # hoop position
        self.dist_threshold = 1.0 / 2  # fixed to check the performance

        # distance between the initial ee position and hoop position
        self.ee2hoop = np.linalg.norm(self.ini_ee_pos - self.hoop_position)

        # specify the action spaces of the acceleration
        self.speed_norm_factor = 10.0e-3
        self.num_speeds = 20 / 2
        # self.speed_range = list(np.linspace(-1, 0, self.num_speeds)[:-1]) + list(np.linspace(0, 1, self.num_speeds))
        self.speed_range = [-1.5, -1.0, -0.5, -0.1, 0, 0.1, 0.5, 1.0, 1.5]
        self.speed_range = np.asarray(self.speed_range) / self.speed_norm_factor
        self.action_spaces = [[0]] * self.num_fixed_joints + \
                             [list(self.speed_range)] * (self.num_joints-self.num_fixed_joints)
        self.action_noise = 1.0 / self.num_speeds / self.speed_norm_factor / 2.0 # random noise added to the action

        # enumerate all possible actions available
        self.action_combinations = np.asarray(list(itertools.product(*self.action_spaces)))
        self.action_idxes = np.arange(self.action_combinations.shape[0])
        # create array for storing state_action_pairs
        self.ext_action_cmbs = np.hstack((np.zeros((len(self.action_combinations), self.state_dimension)),
                                          self.action_combinations))

        self.state_action_idxes = [4, 5, 10, 11, 16, 17]
        self.state_idxes = [4, 5, 10, 11]

        # ball information
        self.gravity = -10.0

        # maximum number of time steps
        self.max_time = 0.1

        # parameters used for training
        self.policy_greedy = 1.0 # change the steepness of the softmax function of the policy object
        self.epsilon = 1.0 / len(self.action_combinations) # parameter for epsilon-greedy policy function
        self.epsilon = 0.75
        self.epsilon_increase = 0.25 # epsilon increase proportion when perfance reaches a threshold
        self.alpha = 1 # controls the relationship between reward and distance to the hoop
        self.max_reward = 1.0
        self.noise_level = 0.2