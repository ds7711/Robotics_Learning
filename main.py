
import numpy as np
from keras.models import load_model
# import matplotlib.pyplot as plt
import robotics_library as rbl
import os.path
import pdb

# read the environment
env_name = "2d"
model_name = env_name + ".h5"
data_name = env_name + ".npz"
env_obj = rbl.env_dict[env_name]

# create an initial object
link_lengthes = [1, 3, 1, 1, 1, 1, 1]
initial_angles = [0, 0, 0, 0, -np.pi/4, 0]
initial_angular_velocities = np.zeros(env_obj.num_joints)

initial_rm = rbl.Robotic_Manipulator_Naive(link_lengthes, initial_angles, initial_angular_velocities)
print(initial_rm.loc_joints())
print(initial_angles)
print(env_obj.hoop_position)

# create the hoop

pdb.set_trace()
# initialize the q_value function object
if os.path.isfile(model_name):
    q_obj = load_model(model_name)
else:
    q_obj = rbl.get_q_func([19, 50, 20, 1])


# train the q_value function object
env_obj.hoop_size = 2
print(env_obj.hoop_size)
q_obj, reward_list, score_list = rbl.neural_fitted_q_algorithm(initial_rm, q_obj, env_obj, num_iterations=10, model_name=model_name)
np.savez_compressed(data_name, reward_list=reward_list,
                    score_list=score_list)
q_obj.save(model_name)