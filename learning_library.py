import numpy as np
import kinematics_library as knl
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
import pdb
import copy


def shaping_training(ini_mover_str, ini_releaser_str, env):

    # initialize score criterion and reward threshold
    reward_threshold = env.ee2hoop - 0.5
    strength_threshold_factor = 1.5

    # initialize the policy object
    ini_mover = get_move_agent(ini_mover_str)
    ini_releaser = get_release_agent(ini_releaser_str)
    policy = PolicyObject(ini_mover, ini_releaser, env, mover_q_ub=0)

    # test the performance of the exploiter
    ini_ra = copy.deepcopy(env.ini_ra)
    rewards, scores = policy.test_policy_performance(ini_ra, move_epislon=0.9,
                                                     release_epislon=0.8,
                                                     threshold=reward_threshold,
                                                     num_test=1000)
    avg_reward = np.mean(rewards)
    avg_score = np.mean(scores)

    # initial avg score
    max_score = avg_score
    avg_score_list = [avg_score]
    reward_freq_list = [np.mean(rewards > 0)]
    reward_thres_list = []
    print("Initial rewards and scores are: %f, %f" % (avg_reward, avg_score))

    # initialize container for trajectories that actually scored!
    score_trj_pool = TrajectoryPool(max_trajectories=100000, env=env)

    # loop until score performance reaches the threshold or pleateu or maximum iteration reached
    for iii in range(10):

        # training parameter
        release_epislon = 0.75
        move_epislon = 0.9
        noise_level = 0.2
        discounting_factor = 0.9
        release_eps_factor = 4

        score_list = []

        # delete the old training data and initialize the new container
        trj_pool = TrajectoryPool(max_trajectories=1e10, env=env)
        num_trjs = 3000

        # Repeatedly generate trajectories using explorer & exploiter

        for _ in range(num_trjs):
            # generate trajectories using explorer, exploiter, and joker
            # generate trj using random explorer
            ra = copy.deepcopy(env.ini_ra)
            states_list, mas_list, ras_list, rewards, \
            score = policy.random_explorer(ra, np.random.randint(1, 100),
                                           reward_threshold, noise=noise_level)
            score_list.append(score)
            trj_pool.add_trj(states_list, mas_list, ras_list, rewards)
            if score:
                score_trj_pool.add_trj(states_list, mas_list, ras_list, rewards)

            # repeat if it's the first time
            if iii == 0:
                for _ in range(2):
                    ra = copy.deepcopy(env.ini_ra)
                    states_list, mas_list, ras_list, rewards, \
                        score = policy.random_explorer(ra, np.random.randint(1, 100),
                                                       reward_threshold, noise=noise_level)
                    score_list.append(score)
                    trj_pool.add_trj(states_list, mas_list, ras_list, rewards)
                    if score:
                        score_trj_pool.add_trj(states_list, mas_list, ras_list, rewards)

            # generate trj using power exploring policy
            if iii != 0:
                ra = copy.deepcopy(env.ini_ra)
                states_list, mas_list, ras_list, rewards, \
                    score = policy.power_exploring_trajectory(ra, move_epislon=move_epislon,
                                                              release_epislon=release_epislon,
                                                              threshold=reward_threshold, noise=noise_level)
                score_list.append(score)
                trj_pool.add_trj(states_list, mas_list, ras_list, rewards)
                if score:
                    score_trj_pool.add_trj(states_list, mas_list, ras_list, rewards)

                # generate try using greedy plus random
                ra = copy.deepcopy(env.ini_ra)
                states_list, mas_list, ras_list, rewards, \
                    score = policy.greedy_plus_random_explorer(ra, move_epislon=move_epislon,
                                                               release_epislon=release_epislon,
                                                               threshold=reward_threshold, noise=noise_level)
                score_list.append(score)
                trj_pool.add_trj(states_list, mas_list, ras_list, rewards)
                if score:
                    score_trj_pool.add_trj(states_list, mas_list, ras_list, rewards)

                # generate trj using power plus random
                ra = copy.deepcopy(env.ini_ra)
                states_list, mas_list, ras_list, rewards, \
                    score = policy.power_plus_random_explorer(ra, move_epislon=move_epislon,
                                                              release_epislon=release_epislon,
                                                              threshold=reward_threshold, noise=noise_level)
                score_list.append(score)
                trj_pool.add_trj(states_list, mas_list, ras_list, rewards)
                if score:
                    score_trj_pool.add_trj(states_list, mas_list, ras_list, rewards)

            print("."),

        # collect good, bad, and exploratory trajectories
        # good: data that has the biggest reward
        # bad: data that has least reward
        # exploratory: data from random behavior
        final_rewards = np.asarray(trj_pool.final_rewards)
        reward_freq_list.append(np.mean(final_rewards > 0))

        release_x, release_y = trj_pool.data4release_agent()
        move_x, move_y = trj_pool.data4move_agent(discounting=discounting_factor)

        # train the releaser and mover
        tmp_releaser = get_release_agent(ini_releaser_str)
        tmp_mover = get_move_agent(ini_mover_str)
        policy = PolicyObject(tmp_mover, tmp_releaser, env, mover_q_ub=0)
        fitted_release_agent = training2converge(policy.releaser_q, release_x, release_y, batch_size=10000,
                                                 epochs=100, verbose=0)
        fitted_move_agent = training2converge(policy.mover_q, move_x, move_y, batch_size=10000, epochs=20, verbose=0)

        # test the training results
        # releaser

        good_release_idx = release_y > 0
        bad_release_idx = release_y < 0
        good_releaser_y = release_y[good_release_idx]
        bad_releaser_y = release_y[bad_release_idx]
        est_release_y = policy.releaser_q.predict(release_x)
        est_good_release_y = est_release_y[good_release_idx]
        est_bad_release_y = est_release_y[bad_release_idx]
        print(np.mean(est_good_release_y > 0), np.mean(est_bad_release_y < 0))

        # mover
        good_move_idx = move_y > 0
        bad_move_idx = move_y < 0
        good_mover_y = move_y[good_move_idx]
        bad_mover_y = move_y[bad_move_idx]
        est_move_y = policy.mover_q.predict(move_x)
        est_good_move_y = est_move_y[good_move_idx]
        est_bad_move_y = est_move_y[bad_move_idx]
        print(np.mean(est_good_move_y > 0), np.mean(est_bad_move_y < 0))

        pdb.set_trace()

        # caculate the q_value criterion for releasing the ball
        good_release_idx = release_y > 0
        good_release_x = release_x[good_release_idx, :]
        q_vals = np.squeeze(policy.releaser_q.predict(good_release_x))
        q_val_threshold = (np.mean(q_vals) + release_eps_factor * np.max(q_vals)) / (1.0 + release_eps_factor)
        release_epislon = q_val_threshold

        # test that exploiter could reliably do the current best behavior and record the data
        ini_ra = copy.deepcopy(env.ini_ra)
        rewards, scores = policy.test_policy_performance(ini_ra, move_epislon=move_epislon,
                                                         release_epislon=release_epislon,
                                                         threshold=reward_threshold)
        avg_reward = np.mean(rewards > 0)
        avg_score = np.mean(scores)
        avg_score_list.append(avg_score)
        print("Iteration: %d, reward threshold: %f, q_val_threshold: %f " % (iii, reward_threshold, q_val_threshold))
        print("Current rewards and scores are: %f, %f" % (avg_reward, avg_score))
        print(reward_freq_list)

        policy.mover_q.save("mover.h5")
        policy.releaser_q.save("releaser.h5")

        reward_thres_list.append(reward_threshold)

        # if yes, increase the reward threshold for next training iteration
        if reward_freq_list[-1] > reward_freq_list[-2]:
            max_score = avg_score
            reward_threshold /= strength_threshold_factor

    return(np.asarray(avg_score_list), np.asarray(reward_freq_list), np.asarray(reward_thres_list))


def get_move_agent(units_list, common_activation_func="tanh", regularization=regularizers.l2(0.1)):
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
    model.compile(loss="mean_squared_error", optimizer="sgd")
    return(model)


def get_release_agent(units_list, common_activation_func="tanh", regularization=regularizers.l2(0.1)):
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
    model.compile(loss="mean_squared_error", optimizer="sgd")
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
    # alpha = 1
    # beta = 1
    dist2t = knl.ball2hoop(pos, vel, target_pos, gravity)

    if dist2t < threshold:
        reward = (alpha / (alpha + dist2t) - 0.5) * 2.0
        # if reward > 0.9:
        #     pdb.set_trace()
        return(reward, dist2t)

    else:
        reward = (beta / (beta + dist2t) - 0.5) * 2.0
        return(reward, dist2t)


class PolicyObject(object):

    def __init__(self, move_agent, release_agent,
                 env, mover_q_ub=0):
        """
        initialize policy object
        :param move_agent:
        :param release_agent:
        :param env:
        :param mover_q_ub: q values higher than the upper bound shouldn't be trusted
        """

        # action spaces
        # self.env = env
        self.action_cmbs = np.copy(env.action_combinations)
        self.ext_action_cmbs = np.copy(env.ext_action_cmbs)
        self.action_idxes = np.copy(env.action_idxes)

        # mover and releaser
        self.mover_q = move_agent
        self.releaser_q = release_agent
        self.hoop_position = env.hoop_position
        self.max_score_dist = env.dist_threshold
        self.mover_q_ub = mover_q_ub

        # state dimension
        self.state_dimension = env.state_dimension
        self.gravity = env.gravity
        self.max_time = env.max_time
        self.state_idxes = env.state_idxes
        self.state_action_idxes = env.state_action_idxes

    def test_move_q(self, ra, move_action_list, release_action_list, threshold):
        """
        test the goodness of the current model
        :param ra:
        :param move_action_list:
        :param release_action_list:
        :param threshold:
        :return:
        """
        state_list = [np.copy(ra.state)]
        reward_list = []
        score = 0
        for iii in range(len(release_action_list)):
            move_action = move_action_list[iii]
            release_action = release_action_list[iii]
            # pdb.set_trace()
            ra.update(move_action, release_action)
            state_list.append(np.copy(ra.state))
            if ra.release:
                pos = ra.loc_joints()[-1][:-1]
                vel = ra.cal_ee_speed()[:-1]
                reward, dist2t = reward_function(pos, vel, threshold, self.hoop_position, self.gravity)
                score = dist2t < self.max_score_dist
            else:
                reward = 0
            reward_list.append(reward)
        state_list = np.asarray(state_list[:-1])
        x = np.hstack((state_list, move_action_list))[:-1, :]
        x = x[:, self.state_action_idxes]
        move_q_vals = self.mover_q.predict(normalize_x(x))
        _, release_q_vals = self._test_release_q(ra, state_list, threshold)
        return(np.asarray(state_list), np.squeeze(move_q_vals), np.squeeze(release_q_vals),
               np.asarray(reward_list), score)

    def _test_release_q(self, ra, final_state_list, threshold):
        reward_list = []
        final_state_list = np.asarray(final_state_list)
        final_state_list = normalize_x(final_state_list)
        for state in final_state_list:
            ra.joint_angles = state[:self.state_dimension/2]
            ra.angular_velocities = state[self.state_dimension/2:]
            pos, vel = ra.loc_joints()[-1][:-1], ra.cal_ee_speed()[:-1]
            reward, dist2t = reward_function(pos, vel, threshold, self.hoop_position, self.gravity)
            reward_list.append(reward)
        release_q_vals = np.squeeze(self.releaser_q.predict(final_state_list[:, self.state_idxes]))
        return(np.asarray(reward_list), release_q_vals)

    def _random_move(self, release_epislon):
        """
        randomly select a move action, select release based on epsilon
        :param release_epislon:
        :return:
        """
        rdx = np.random.choice(self.action_idxes)  # randomly choose a move action
        move_action = self.action_cmbs[rdx]
        release_action = np.random.rand() > release_epislon
        return (np.copy(move_action), release_action)

    def _epsilon_greedy_action(self, state, move_epislon, release_epislon):
        """
        select move and release action
        :param state: joint angles and velocities
        :param move_epislon:
        :param release_epislon:
        :return:
        """
        self.ext_action_cmbs[:, :self.state_dimension] = state
        ext_cmbs_2d = self.ext_action_cmbs[:, self.state_action_idxes]

        mover_q_values = self.mover_q.predict(normalize_x(ext_cmbs_2d))
        releaser_q_values = np.squeeze(self.releaser_q.predict(normalize_x(state[np.newaxis,
                                                                           self.state_idxes])))
        # pdb.set_trace()
        if np.random.rand() < move_epislon:
            if np.random.randn() < move_epislon:
                # if predicted q values from nn is larger than the biggest q values from the training data, discard it
                mover_q_values[mover_q_values > self.mover_q_ub] = 0
            act_idx = np.argmax(mover_q_values)
            move_action = self.action_cmbs[act_idx]
        else:
            act_idx = np.random.choice(self.action_idxes)
            move_action = self.action_cmbs[act_idx]

        release_probability = (releaser_q_values + 1) / 2.0
        if release_probability > release_epislon:
            release_action = 1
        # elif release_probability < 0.5:
        #     release_action = np.random.rand() < 1e-2
        else:  # when confidence is low, don't throw the ball
            release_action = np.random.rand() < release_probability / 20
        return(np.copy(move_action), release_action)

    def random_explorer(self, ra, num_movements, threshold, noise):
        """
        select actions randomly
        :param ra:
        :param num_movements: select the number of movements before releasing the ball
        :param threshold:
        :param noise:
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
        score = 0

        for release_action in release_action_lsit:
            # return reward based on whether the ball was released
            if not release_action:  # first test whether the ball was released
                reward = 0.0
                # get the action randomly, never select release until number of movements were tried
                move_action, _ = self._random_move(2)  # set release_epislon to be bigger than 1 so that no release

                # add some noise to the data so no repeated trajectories
                added_noise = np.random.randn(len(move_action)) * np.abs(move_action) * noise
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
                reward, dist2t = reward_function(ee_pos, ee_speed, threshold, hoop_position, gravity)
                score = dist2t < self.max_score_dist
                move_action = np.zeros(self.action_cmbs.shape[1])
                move_action_list.append(move_action)  # add a action selected at the last state
            reward_list.append(reward)

        return (np.asarray(state_list), np.asarray(move_action_list),
                release_action_lsit,
                np.asarray(reward_list), score)

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
        score = 0

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
                reward, dist2t = reward_function(ee_pos, ee_speed, threshold, self.hoop_position,
                                                 self.gravity)
                score = dist2t < self.max_score_dist
                move_action = np.zeros(self.action_cmbs.shape[1])
                move_action_list.append(move_action)
                ra.release = True  # update the ra state to break the loop

            release_action_lsit.append(release_action)
            reward_list.append(reward)
            # print(ra.time)

        if ra.time > self.max_time:  # if the ball was not released, force it to be released so that we get some data
            move_action, release_action = np.zeros(self.action_cmbs.shape[1]), 1
            ee_pos = ra.loc_joints()[-1][:-1]
            ee_speed = ra.cal_ee_speed()[:-1]
            reward, dist2t = reward_function(ee_pos, ee_speed, threshold, self.hoop_position,
                                             self.gravity)
            score = dist2t < self.max_score_dist
            move_action_list.append(move_action)
            release_action_lsit.append(release_action)
            reward_list.append(reward)

        return (np.asarray(state_list), np.asarray(move_action_list),
                np.asarray(release_action_lsit),
                np.asarray(reward_list), score)

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
        score = 0
        # pdb.set_trace()
        while (not ra.release) and ra.time < self.max_time:
            # get the action randomly, never select release until number of movements were tried
            move_action, release_action = self._epsilon_greedy_action(ra.state, move_epislon,
                                                                      release_epislon)
            if not release_action:
                # add noise to the actions
                added_noise = np.random.randn(self.action_cmbs.shape[1]) * noise * np.abs(move_action)
                move_action += added_noise
                # store the action
                move_action_list.append(move_action)
                # update the robot
                ra.update(move_action, release_action)

                # add the new state to the state list
                state_list.append(np.copy(ra.state))
                reward = 0.0
            else:
                # return reward based on where the ball was released
                ee_pos = ra.loc_joints()[-1][:-1]
                ee_speed = ra.cal_ee_speed()[:-1]
                reward, dist2t = reward_function(ee_pos, ee_speed, threshold, self.hoop_position,
                                                 self.gravity)
                score = dist2t < self.max_score_dist
                move_action = np.zeros(self.action_cmbs.shape[1])
                move_action_list.append(move_action)
                ra.release = True  # release the ball

            # print(ra.time)
            reward_list.append(reward)
            release_action_lsit.append(release_action)

        if ra.time > self.max_time:  # if the ball was not released, force it to be released so that we get some data
            move_action, release_action = np.zeros(self.action_cmbs.shape[1]), 1
            ee_pos = ra.loc_joints()[-1][:-1]
            ee_speed = ra.cal_ee_speed()[:-1]
            reward, dist2t = reward_function(ee_pos, ee_speed, threshold, self.hoop_position,
                                             self.gravity)
            score = dist2t < self.max_score_dist
            move_action_list.append(move_action)
            release_action_lsit.append(release_action)
            reward_list.append(reward)

        return (np.asarray(state_list), np.asarray(move_action_list),
                np.asarray(release_action_lsit),
                np.asarray(reward_list), score)

    def greedy_plus_random_explorer(self, ra, move_epislon, release_epislon, threshold, noise):
        """
        create new trajectories by extending beyond current best policies
        :param ra:
        :param move_epislon:
        :param release_epislon:
        :param threshold:
        :return:
        """

        states0, move_actions0, release_actions0, rewards0, _ = self.epsilon_greedy_trajectory(ra, move_epislon,
                                                                                               release_epislon,
                                                                                               threshold)
        # release_actions0[-1] = 0
        num_extra_movements = np.random.randint(2, len(states0)+2)
        ra.release = 0
        states1, move_actions1, release_actions1, rewards1, score = self.random_explorer(ra, num_extra_movements,
                                                                                         threshold, noise)
        state_list = np.vstack((states0[:-1], states1))
        move_action_list = np.vstack((move_actions0[:-1], move_actions1))
        release_action_list = np.concatenate((release_actions0[:-1], release_actions1))
        reward_list = np.concatenate((rewards0[:-1], rewards1))

        return (np.asarray(state_list), np.asarray(move_action_list),
                np.asarray(release_action_list),
                np.asarray(reward_list), score)

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

        states0, move_actions0, release_actions0, rewards0, _ = self.power_exploring_trajectory(ra, move_epislon,
                                                                                                release_epislon,
                                                                                                threshold, noise)
        # release_actions0[-1] = 0
        num_extra_movements = np.random.randint(2, len(states0)+2)
        ra.release = 0
        states1, move_actions1, release_actions1, rewards1, score = self.random_explorer(ra, num_extra_movements,
                                                                                         threshold, noise)
        state_list = np.vstack((states0[:-1], states1))
        move_action_list = np.vstack((move_actions0[:-1], move_actions1))
        release_action_list = np.concatenate((release_actions0[:-1], release_actions1))
        reward_list = np.concatenate((rewards0[:-1], rewards1))

        return (np.asarray(state_list), np.asarray(move_action_list),
                np.asarray(release_action_list),
                np.asarray(reward_list), score)

    def test_policy_performance(self, ini_ra, move_epislon, release_epislon, threshold, num_test=1000):
        score_list = []
        reward_list = []
        for iii in range(num_test):
            ra = copy.deepcopy(ini_ra)
            states, mas, ras, rewards, score = self.epsilon_greedy_trajectory(ra, move_epislon, release_epislon,
                                                                              threshold)
            score_list.append(score)
            reward_list.append(rewards[-1])
        return(np.asarray(reward_list), np.asarray(score_list))


class TrajectoryPool(object):
    """
    define an object to store trajectories
    """
    def __init__(self, max_trajectories=1e8, env=None):
        """
        initialize the data pool for storing data
        :param max_trajectories:
        :param env:
        """
        self.states_list = []
        self.move_actions_list = []
        self.release_actions_list = []
        self.rewards_list = []
        self.final_rewards = []
        self.max_trajectories = max_trajectories
        self.num_trajectories = 0
        self.good_idxes = []
        self.bad_idxes = []
        self.netral_idxes = []
        self.trj_sep_bd = [0, 0]
        self.final_state_list = []

        # count the good data pairs and bad data pairs
        self.num_good_xy = 0
        self.num_bad_xy = 0
        self.num_neutral_xy = 0

        self.state_idxes = env.state_idxes
        self.state_action_idxes = env.state_action_idxes

    def _add_trj(self, states, move_actions, release_actions, rewards):
        """
        add trajectories to the data pool
        :param states:
        :param move_actions:
        :param release_actions:
        :param rewards:
        :return:
        """
        self.states_list.append(states)
        self.move_actions_list.append(move_actions)
        self.release_actions_list.append(release_actions)
        self.rewards_list.append(rewards)
        self.final_rewards.append(rewards[-1])
        self.final_state_list.append(states[-1])

    def add_trj(self, states, move_actions, release_actions, rewards):
        if self.num_trajectories < self.max_trajectories:
            self._add_trj(states, move_actions, release_actions, rewards)

            if rewards[-1] > self.trj_sep_bd[1]:
                self.good_idxes.append(copy.copy(self.num_trajectories))
                self.num_good_xy += len(states) - 1
            elif rewards[-1] < self.trj_sep_bd[0]:
                self.bad_idxes.append(copy.copy(self.num_trajectories))
                self.num_bad_xy += len(states) - 1
            else:
                self.netral_idxes.append(copy.copy(self.num_trajectories))
                self.num_neutral_xy += len(states) - 1
            self.num_trajectories += 1

        else:
            print("No capacity left!")
            pdb.set_trace()

    def _good_bad_ratio(self, type="release"):
        """
        calculate the relative ratio of good and bad examples
        :return:
        """
        if type == "release":
            num_bad = len(self.bad_idxes)
            num_good = len(self.good_idxes)
            if num_bad > num_good:
                ratio = int(num_bad / num_good + 0.5)
                return(ratio, 1)
            else:
                ratio = int(num_good / num_bad)
                return(1, ratio)
        if type == "move":
            num_bad = self.num_bad_xy
            num_good = self.num_good_xy
            if num_bad > num_good:
                ratio = int(num_bad / num_good + 0.5)
                return(ratio, 1)
            else:
                ratio = int(num_good / num_bad)
                return(1, ratio)

    @staticmethod
    def _propogate_rewards(rewards, discounting):
        """
        propagate the rewards to the previous steps
        :param rewards:
        :param discounting:
        :return:
        """
        num_rewards = len(rewards)
        new_rewards = np.copy(rewards)
        for iii in range(0, num_rewards-1):
            new_rewards[num_rewards-iii-2] = new_rewards[num_rewards-iii-1] * discounting
        return(new_rewards)

    def update(self, mover_q=None, reward_threshold=None, trj_sep_bd=None, discounting=1.0):
        pass

    def data4release_agent(self):
        X = np.asarray(self.final_state_list)
        Y = np.asarray(self.final_rewards)
        good_X = X[self.good_idxes, :]
        good_X = good_X[:, self.state_idxes]
        good_Y = Y[self.good_idxes]

        bad_X = X[self.bad_idxes, :]
        bad_X = bad_X[:, self.state_idxes]
        bad_Y = Y[self.bad_idxes]

        neutral_X = X[self.netral_idxes, :]
        neutral_X = neutral_X[:, self.state_idxes]
        neutral_Y = Y[self.netral_idxes]

        num_good, num_bad = self._good_bad_ratio(type="release")
        if num_good > 1:
            good_X = np.vstack([good_X for _ in range(num_good)])
            good_Y = np.concatenate([good_Y for _ in range(num_good)])
        if num_bad > 1:
            bad_X = np.vstack([bad_X for _ in range(num_bad)])
            bad_Y = np.concatenate([bad_Y for _ in range(num_bad)])

        raw_X = np.vstack((good_X, bad_X, neutral_X))
        X = normalize_x(raw_X)
        raw_Y = np.concatenate((good_Y, bad_Y, neutral_Y))
        Y = normolize_y(raw_Y)

        return(X, Y)

    def _combine_trj(self, idxes, discounting):
        if len(idxes) == 0:
            X = np.empty((0, len(self.state_action_idxes)))
            Y = np.empty((0))
        else:
            states_list = []
            actions_list = []
            Y = []
            for iii in idxes:
                tmp_states = self.states_list[iii][:-1]
                tmp_actions = self.move_actions_list[iii][:-1]
                states_list.append(tmp_states)
                actions_list.append(tmp_actions)
                tmp_rewards = self.rewards_list[iii]
                tmp_rewards = self._propogate_rewards(tmp_rewards, discounting)
                Y.append(tmp_rewards[1:])
            states_list = np.vstack(states_list)
            actions_list = np.vstack(actions_list)
            X = np.hstack((states_list, actions_list))
            Y = np.concatenate(Y)
        return(X[:, self.state_action_idxes], Y)

    def data4move_agent(self, discounting):
        good_X, good_Y = self._combine_trj(self.good_idxes, discounting)
        bad_X, bad_Y = self._combine_trj(self.bad_idxes, discounting)
        neutral_X, neutral_Y = self._combine_trj(self.netral_idxes, discounting)

        num_good, num_bad = self._good_bad_ratio(type="move")
        if num_good > 1:
            good_X = np.vstack([good_X for _ in range(num_good)])
            good_Y = np.concatenate([good_Y for _ in range(num_good)])
        if num_bad > 1:
            bad_X = np.vstack([bad_X for _ in range(num_bad)])
            bad_Y = np.concatenate([bad_Y for _ in range(num_bad)])

        raw_X = np.vstack((good_X, bad_X, neutral_X))
        X = normalize_x(raw_X)
        raw_Y = np.concatenate((good_Y, bad_Y, neutral_Y))
        Y = normolize_y(raw_Y)

        return(X, Y)


def normalize_x(old_x, time_step=1e-3):
    """
    normalize the x so that they are relatively on the same scale
    :param old_x:
    :param time_step:
    :return:
    """
    signs = np.sign(old_x)
    abs_x = np.abs(old_x) / time_step
    log_x = np.log(1.0 + abs_x)
    x = signs * log_x
    return(x)


def normolize_y(old_y):
    """
    rescale the y so that gradient descent can work, y values around 0 has strange gradient
    :param old_y:
    :return:
    """
    avg = np.mean(np.abs(old_y))
    new_y = old_y / avg
    new_y = np.clip(new_y, -1.0, 1.0)
    # return(old_y)
    return(new_y)


def training2converge(agent, x, y, batch_size=10000, epochs=100, count_threshold=1, verbose=0):
    """
    training the agent until cross-validation accuracy doesn't decrease
    :param agent:
    :param x:
    :param y:
    :param batch_size:
    :param epochs:
    :param count_threshold:
    :param verbose:
    :return:
    """
    training_flag = True
    minimum_val_lost = np.inf
    count = 0
    while training_flag:
        ra_hist = agent.fit(x, y[:, np.newaxis], batch_size=batch_size, epochs=epochs,
                            validation_split=0.3, verbose=verbose)
        val_loss = np.mean(ra_hist.history["val_loss"])
        print("%f......." % val_loss),
        if minimum_val_lost - val_loss > 0.005:
            minimum_val_lost = val_loss
        else:
            count += 1
        if count >= count_threshold:
            training_flag = False
            print("Training completed! Validation loss stopped decreasing. ")
    return(agent)

