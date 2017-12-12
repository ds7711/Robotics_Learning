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


def random_explorer(num_movements, threshold, env):
    """
    an explorer that explores certain number of steps until release happened
    :param num_movements: 
    :param threshold: 
    :param env: 
    :return: 
    """

    # initialize the robot arm
    ra = copy.deepcopy(env.ini_ra)

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
    reward_list = [0.0]

    for _ in range(num_movements):
        # get the action randomly, never select release until number of movements were tried
        move_action, release_action = _random_move(2, action_cmbs, action_idxes)
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
    release_action_lsit[-1] = release_action

    return(np.asarray(state_list), np.asarray(move_action_list),
           release_action_lsit,
           np.asarray(reward_list))


def hybrid_move():
    pass


# class PolicyObject(object):
#
#     def __init__(self, move_agent, release_agent,
#                  action_cmbs, ext_action_cmbs,
#                  state_dimension):
#
#         self.action_combinations = action_cmbs
#         self.ext_action_cmbs = ext_action_cmbs
#         self.action_indexes = np.arange(action_cmbs.shape[0])
#         self.mover_q = move_agent
#         self.releaser_q = release_agent
#         self.state_dimension = state_dimension
#
#     def select_action(self, state, policy_type=None):
#         """
#         return action based on the state
#         :param state: joint angle and velocity
#         :param policy_type:
#         :return:
#         """
#         if policy_type == "softmax":
#             action = self.softmax_policy(state)
#         elif policy_type == "epsilon_greedy":
#             action = self.epsilon_greedy_policy(state)
#         else:
#             print("Errors in policy type of the reward function!!!")
#         return(action)
#
#     def epsilon_greedy_policy(self, state):
#         """
#         return the selected action based on the epsilon greedy
#         :param state:
#         :return:
#         """
#         self.ext_action_cmbs[:, :self.state_dimension] = state
#         two_d_data = self.ext_action_cmbs[:, :[4, 5, 10, 11]]
#
#         mover_q_values = self.mover_q.predict(self.ext_action_cmbs)
#         releaser_q_values = np.squeeze(self.releaser_q.predict(state[np.newaxis, :]))
#
#         est_q_values = self.q_obj.predict(self.env_obj.ext_action_cmbs)
#         if np.random.rand() < self.env_obj.epsilon:
#             act_idx = np.argmax(est_q_values)
#             return(self.env_obj.action_combinations[act_idx])
#         else:
#             act_idx = np.random.choice(self.action_indexes)
#             return (self.env_obj.action_combinations[act_idx])


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
#


def generate_trajectories(ra, throw_agent, release_agent):
    pass


# supplementary function


def _random_move(epsilon, action_cmbs, action_idxes):
    """
    randomly select a move action, select release based on epsilon
    :param epsilon: 
    :param action_cmbs: 
    :param action_idxes: 
    :return: 
    """
    rdx = np.random.choice(action_idxes)  # randomly choose a move action
    move_action = action_cmbs[rdx]
    release_action = np.random.rand() > epsilon
    return(move_action, release_action)
