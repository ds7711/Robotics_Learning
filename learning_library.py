import numpy as np
import kinematics_library as knl
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
import copy


def shaping_training(env):

    # initialize mover and releaser

    # initialize score criterion and reward threshold
    reward_threshold = None
    threshold_increase = 1.5
    score_threhold = env.score_threshold

    # initialize the exploiter, explorer, and joker
    #   joker choose totally random actions and didn't care about rewards
    #   exploiter uses the current best policy
    #   explorer explore based on the current best policy

    # test the performance of the exploiter

    # initialize container for trajectories that actually scored!

    # loop until score performance reaches the threshold or pleateu or maximum iteration reached
    while True:

        # delete the old training data and initialize the new container

        # Repeatedly generate trajectories using explorer & exploiter
        while True:
            # generate trajectories using explorer, exploiter, and joker

            # collect good, bad, and exploratory trajectories
            # good: data that has the biggest reward
            # bad: data that has least reward
            # exploratory: data from random behavior

            pass

        # Iteratively train the exploiter
        while True:
            # train the exploiter with the data in back-propagating way
            #   1st: train with data of the last time step from all trajectories until converge

            #   2nd: update the q_value from the previous step and then train until converge

            #   3rd: repeat until data from prevous time step doesn't have enough data

            #   4th: train with all the data until converge

            pass

        # test that exploiter could reliably do the current best behavior and record the data

        # increase the reward threshold for next training iteration
        reward_threshold /= threshold_increase

        pass

    # return the exploiter that achieves desiered performance: q function, policy parameters

    pass


def get_move_agent(units_list, common_activation_func="relu", regularization=regularizers.l2(0.1)):
    """
    create a neural network model based on the units_list
    :param units_list: list of integers that specify the number of units in each layer
        The list has to contain at least 3 items.
    :param common_activation_func:
    :param regularization:
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
    model.add(Activation("tanh"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return(model)


def get_release_agent(units_list, common_activation_func="relu", regularization=regularizers.l2(0.1)):
    """
    create a neural network object for classification
    :param units_list:
    :param common_activation_func:
    :param regularization:
    :return:
    """
    input_dimension = units_list[0]
    model = Sequential()
    model.add(Dense(units_list[1], kernel_regularizer=regularization, input_dim=input_dimension))
    model.add(Activation(common_activation_func))
    for i, num_unit in enumerate(units_list[2:-1]):
        model.add(Dense(num_unit, kernel_regularizer=regularization))
        model.add(Activation(common_activation_func))
    model.add(Dense(units_list[-1], kernel_regularizer=regularization))
    model.add(Activation("tanh"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return(model)


def reward_function(pos, vel, threshold, target_pos, gravity):
    """
    calculate the reward at each time step.
    The current reward function is relatively smooth, which may be a good thing or a bad thing!!!
    :param pos: position of each effector
    :param vel:
    :param threshold:
    :param target_pos:
    :param gravity:
    :return:
    """
    alpha = threshold
    beta = threshold
    dist2t = knl.ball2hoop(pos, vel, target_pos, gravity)

    if dist2t < threshold:
        reward = alpha / (alpha + dist2t) - 0.5
        return(reward * 2.0)

    else:
        reward = beta / (beta + dist2t) - 0.5
        return(reward * 2.0)


def random_explorer(ra, num_movements, threshold, env):
    """
    an explorer that explores certain number of steps until release happened
    :param num_movements: 
    :param threshold: 
    :param env: 
    :return: 
    """

    # get hoop position
    hoop_position = env.hoop_position
    gravity = env.gravity

    # get values needed
    action_cmbs = env.action_combinations
    # ext_action_cmbs = env.ext_action_cmbs
    action_idxes = env.action_idxes

    # initialize list to store the trajectory
    state_list = [np.copy(ra.state)]
    move_action_list = []
    release_action_lsit = np.zeros(num_movements)
    release_action_lsit[-1] = 1
    reward_list = [0.0]

    for release_action in release_action_lsit:
        # get the action randomly, never select release until number of movements were tried
        move_action, _ = _random_move(2, action_cmbs, action_idxes)
        # store the action
        move_action_list.append(move_action)
        # update the robot
        ra.update(move_action, release_action)

        # add the new state to the state list
        state_list.append(np.copy(ra.state))

        # return reward based on whether the ball was released
        if not release_action:
            reward = 0.0
        else:
            ee_pos = ra.loc_joints()[-1][:-1]
            ee_speed = ra.cal_ee_speed()[:-1]
            reward = reward_function(ee_pos, ee_speed, threshold, hoop_position, gravity)
        reward_list.append(reward)
    move_action, release_action = _random_move(-1, action_cmbs, action_idxes)
    move_action_list.append(move_action)  # add a action selected at the last state

    return(np.asarray(state_list), np.asarray(move_action_list),
           release_action_lsit,
           np.asarray(reward_list))


class PolicyObject(object):

    def __init__(self, move_agent, release_agent,
                 env):
        """
        initialize policy object
        :param move_agent:
        :param release_agent:
        :param env:
        """

        # action spaces
        # self.env = env
        self.action_cmbs = env.action_combinations
        self.ext_action_cmbs = env.ext_action_cmbs
        self.action_idxes = env.action_idxes

        # mover and releaser
        self.mover_q = move_agent
        self.releaser_q = release_agent
        self.hoop_position = env.hoop_position

        # state dimension
        self.state_dimension = env.state_dimension
        self.gravity = env.gravity
        self.max_time = env.max_time

        # epsilon
        # self.exploit_epislon = exploit_epislon
        # self.explore_epislon = explore_epislon

    def _random_move(self, release_epislon):
        """
        randomly select a move action, select release based on epsilon
        :param release_epislon:
        :return:
        """
        rdx = np.random.choice(self.action_idxes)  # randomly choose a move action
        move_action = self.action_cmbs[rdx]
        release_action = np.random.rand() > release_epislon
        return (move_action, release_action)

    def _epsilon_greedy_action(self, state, move_epislon, release_epislon):
        """
        select move and release action
        :param state: joint angles and velocities
        :param move_epislon:
        :param release_epislon:
        :return:
        """

        self.ext_action_cmbs[:, :self.state_dimension] = state
        ext_cmbs_2d = self.ext_action_cmbs[:, [4, 5, 10, 11, 16, 17]]

        mover_q_values = self.mover_q.predict(ext_cmbs_2d)
        releaser_q_values = np.squeeze(self.releaser_q.predict(state[np.newaxis,
                                                                     [4, 5, 10, 11]]))

        if np.random.rand() < move_epislon:
            act_idx = np.argmax(mover_q_values)
            move_action = self.action_cmbs[act_idx]
        else:
            act_idx = np.random.choice(self.action_idxes)
            move_action = self.action_cmbs[act_idx]

        release_action = releaser_q_values > release_epislon

        return(move_action, release_action)

    def random_explorer(self, ra, num_movements, threshold):
        """
        select actions randomly
        :param ra:
        :param num_movements: select the number of movements before releasing the ball
        :param threshold:
        :return:
        """

        # get hoop position
        hoop_position = self.hoop_position
        gravity = self.gravity

        # initialize list to store the trajectory
        state_list = [np.copy(ra.state)]
        move_action_list = []
        release_action_lsit = np.zeros(num_movements)
        release_action_lsit[-1] = 1
        reward_list = []

        for release_action in release_action_lsit:
            # return reward based on whether the ball was released
            if not release_action: # first test whether the ball was released
                reward = 0.0
                # get the action randomly, never select release until number of movements were tried
                move_action, _ = self._random_move(2)  # set release_epislon to be bigger than 1 so that no release

                # add some noise to the data so no repeated trajectories
                added_noise = np.random.randn(len(move_action)) * np.abs(move_action) * 0.3
                move_action += added_noise
                # store the action
                move_action_list.append(move_action)
                # update the robot
                ra.update(move_action, release_action)

                # add the new state to the state list
                state_list.append(np.copy(ra.state))
            else:
                ee_pos = ra.loc_joints()[-1][:-1]
                ee_speed = ra.cal_ee_speed()[:-1]
                reward = reward_function(ee_pos, ee_speed, threshold, hoop_position, gravity)

                move_action = np.zeros(self.action_cmbs.shape[1])
                move_action_list.append(move_action)  # add a action selected at the last state
            reward_list.append(reward)

        return (np.asarray(state_list), np.asarray(move_action_list),
                release_action_lsit,
                np.asarray(reward_list))

    def epsilon_greedy_trajectory(self, ra, move_epislon, release_epislon, threshold):
        """
        generate trajectories based on epsilon greedy method
        :param ra:
        :param move_epislon:
        :param release_epislon:
        :param threshold:
        :return:
        """
        # initialize list to store the trajectory
        state_list = [np.copy(ra.state)]
        move_action_list = []
        release_action_lsit = []
        reward_list = []

        while (not ra.release) and ra.time < self.max_time:
            # get the action randomly, never select release until number of movements were tried
            move_action, release_action = self._epsilon_greedy_action(ra.state, move_epislon,
                                                                      release_epislon)

            if not release_action:
                # store the action
                move_action_list.append(move_action)
                # update the robot
                ra.update(move_action, release_action)

                # add the new state to the state list
                state_list.append(np.copy(ra.state))
                reward = 0.0
            else:
                # return reward based on whether the ball was released
                ee_pos = ra.loc_joints()[-1][:-1]
                ee_speed = ra.cal_ee_speed()[:-1]
                reward = reward_function(ee_pos, ee_speed, threshold, self.hoop_position,
                                         self.gravity)
                move_action = np.zeros(self.action_cmbs.shape[1])
                move_action_list.append(move_action)

            release_action_lsit.append(release_action)
            reward_list.append(reward)

        if ra.time > self.max_time: # if the ball was not released, force it to be released so that we get some data
            move_action, release_action = np.zeros(self.action_cmbs.shape[1]), 1
            ee_pos = ra.loc_joints()[-1][:-1]
            ee_speed = ra.cal_ee_speed()[:-1]
            reward = reward_function(ee_pos, ee_speed, threshold, self.hoop_position,
                                     self.gravity)
            move_action_list.append(move_action)
            release_action_lsit.append(release_action)
            reward_list.append(reward)

        return (np.asarray(state_list), np.asarray(move_action_list),
                np.asarray(release_action_lsit),
                np.asarray(reward_list))

    def power_exploring_trajectory(self, ra, move_epislon, release_epislon, threshold, noise):
        """
        explore new trajectories by adding noise into the existing good trajectories
        :param ra:
        :param move_epislon:
        :param release_epislon:
        :param threshold:
        :param noise:
        :return:
        """
        # initialize list to store the trajectory
        state_list = [np.copy(ra.state)]
        move_action_list = []
        release_action_lsit = []
        reward_list = []

        while (not ra.release) and ra.time < self.max_time:
            # get the action randomly, never select release until number of movements were tried
            move_action, release_action = self._epsilon_greedy_action(ra.state, move_epislon,
                                                                      release_epislon)

            # return reward based on whether the ball was released
            if not release_action:
                reward = 0.0
                # add noise to the actions
                added_noise = np.random.randn(self.action_cmbs.shape[1]) * noise * np.abs(move_action)
                move_action += added_noise
                # store the action
                move_action_list.append(move_action)

                # update the robot
                ra.update(move_action, release_action)

                # add the new state to the state list
                state_list.append(np.copy(ra.state))
            else:
                ee_pos = ra.loc_joints()[-1][:-1]
                ee_speed = ra.cal_ee_speed()[:-1]
                reward = reward_function(ee_pos, ee_speed, threshold, self.hoop_position,
                                         self.gravity)
                move_action = np.zeros(self.action_cmbs.shape[1])
                move_action_list.append(move_action)

            reward_list.append(reward)
            release_action_lsit.append(release_action)

        if ra.time > self.max_time: # if the ball was not released, force it to be released so that we get some data
            move_action, release_action = np.zeros(self.action_cmbs.shape[1]), 1
            ee_pos = ra.loc_joints()[-1][:-1]
            ee_speed = ra.cal_ee_speed()[:-1]
            reward = reward_function(ee_pos, ee_speed, threshold, self.hoop_position,
                                     self.gravity)
            move_action_list.append(move_action)
            release_action_lsit.append(release_action)
            reward_list.append(reward)

        return (np.asarray(state_list), np.asarray(move_action_list),
                np.asarray(release_action_lsit),
                np.asarray(reward_list))

    def greedy_plus_random_explorer(self, ra, move_epislon, release_epislon, threshold):
        """
        create new trajectories by extending beyond current best policies
        :param ra:
        :param move_epislon:
        :param release_epislon:
        :param threshold:
        :return:
        """

        states0, move_actions0, release_actions0, rewards0 = self.epsilon_greedy_trajectory(ra, move_epislon,
                                                                                            release_epislon,
                                                                                            threshold)
        # release_actions0[-1] = 0
        num_extra_movements = np.random.randint(1, len(states0)+1)
        ra.release = 0
        states1, move_actions1, release_actions1, rewards1 = self.random_explorer(ra, num_extra_movements, threshold)
        state_list = np.vstack((states0[:-1], states1))
        move_action_list = np.vstack((move_actions0[:-1], move_actions1))
        release_action_list = np.concatenate((release_actions0[:-1], release_actions1))
        reward_list = np.concatenate((rewards0[:-1], rewards1))

        return (np.asarray(state_list), np.asarray(move_action_list),
                np.asarray(release_action_list),
                np.asarray(reward_list))

    def power_plus_random_explorer(self, ra, move_epislon, release_epislon, threshold, noise):
        """
        combinng power with random exploration to explore better policies
        :param ra:
        :param move_epislon:
        :param release_epislon:
        :param threshold:
        :param noise:
        :return:
        """

        states0, move_actions0, release_actions0, rewards0 = self.power_exploring_trajectory(ra, move_epislon,
                                                                                             release_epislon,
                                                                                             threshold, noise)
        # release_actions0[-1] = 0
        num_extra_movements = np.random.randint(1, len(states0)+1)
        ra.release = 0
        states1, move_actions1, release_actions1, rewards1 = self.random_explorer(ra, num_extra_movements, threshold)
        state_list = np.vstack((states0[:-1], states1))
        move_action_list = np.vstack((move_actions0[:-1], move_actions1))
        release_action_list = np.concatenate((release_actions0[:-1], release_actions1))
        reward_list = np.concatenate((rewards0[:-1], rewards1))

        return (np.asarray(state_list), np.asarray(move_action_list),
                np.asarray(release_action_list),
                np.asarray(reward_list))

#     def softmax_policy(self, state_action, ):
#         """
#         choose action based on the probability from softmax of the q value
#         :param state:
#         :param exploring_factor: balance exploration and exploitation
#         :return:
#         """
#         self.ext_action_cmbs[:, :self.env_obj.state_dimension] = state
#         est_q_values = self.q_obj.predict(self.env_obj.ext_action_cmbs)
#         exponential_values = np.exp(est_q_values * self.greedy)
#         probs = exponential_values / np.sum(exponential_values)
#         action = self.env_obj.action_combinations[np.random.choice(self.action_indexes, p=np.squeeze(probs))]
#         return(action)


# supplementary function

