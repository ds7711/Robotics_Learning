
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
env_obj = rbl.Env2D()

# create an initial object
link_lengthes = [1, 3, 1, 1, 1, 1, 1]
initial_angles = [0, 0, 0, 0, -np.pi/4, 0]
initial_angular_velocities = np.zeros(env_obj.num_joints)

initial_rm = rbl.Robotic_Manipulator_Naive(link_lengthes, initial_angles, initial_angular_velocities)
print(initial_rm.loc_joints()[-1])
# print(initial_angles)
print(env_obj.hoop_position)

# create the hoop

# pdb.set_trace()
# initialize the q_value function object
if os.path.isfile(model_name):
    q_obj = load_model(model_name)
else:
    q_obj = rbl.get_q_func([19, 50, 20, 1])

# q_obj = rbl.get_q_func([19, 30, 15, 1])


# train the q_value function object
positive_data = rbl.DataPool(q_obj, max_trajectories=100)

env_obj.hoop_size = 4.0
print(env_obj.hoop_size)
q_obj, reward_list, score_list = rbl.shaping_training(q_obj, env_obj, positive_data,
                                                      num_iterations=50, model_name=model_name,
                                                      policy_type="epsilon_greedy")
np.savez_compressed(data_name, reward_list=reward_list,
                    score_list=score_list)
q_obj.save(model_name)
