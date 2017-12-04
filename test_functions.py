
import numpy as np
from keras.models import load_model
# import matplotlib.pyplot as plt
import robotics_library as rbl

# read the environment
env_name = "shaping"
model_name = env_name + ".h5"
env_obj = rbl.env_dict[env_name]

# create an initial object
link_lengthes = [1, 3, 1, 1, 1, 1, 1]
initial_angles = [0, 0, 0, 0, np.pi, 0]
initial_angular_velocities = np.zeros(env_obj.num_joints)

initial_rm = rbl.Robotic_Manipulator_Naive(link_lengthes, initial_angles, initial_angular_velocities)
print(initial_rm.loc_joints())

# create the hoop

# initialize the q_value function object
q_obj = rbl.get_q_func([19, 50, 20,  1])
q_obj = load_model(model_name)


# train the q_value function object
q_obj, reward_list, score_list = rbl.neural_fitted_q_algorithm(initial_rm, q_obj, env_obj, num_iterations=100, model_name=model_name)
# q_obj.save(model_name)
