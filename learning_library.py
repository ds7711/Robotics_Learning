import numpy as np
import kinematics_library as knl
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers


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
    model.add(Activation("sigmoid"))
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

def generate_trajectories(ra, throw_agent, release_agent):
    pass
