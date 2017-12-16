import numpy as np
import pdb
import kinematics_library as knl
import learning_library as lnl
import environment
# import matplotlib.pylab as plt
import copy
import IPython


# initialize objects
env = environment.Env2D()
ra = copy.deepcopy(env.ini_ra)
release_agent = lnl.get_release_agent([4, 30, 10, 1])
move_agent = lnl.get_move_agent([6, 50, 20, 1])
policy = lnl.PolicyObject(move_agent, release_agent, env)


# 7. debug for the TrajectoryPool
trj_pool = lnl.TrajectoryPool(max_trajectories=10000, env=env)

for iii in range(1000):
    ra = copy.deepcopy(env.ini_ra)
    states_list, mas_list, ras_list, rewards_list = policy.random_explorer(ra, 20, 7)
    trj_pool.add_trj(states_list, mas_list, ras_list, rewards_list)
    pass

release_x, release_y = trj_pool.data4release_agent()
move_x, move_y = trj_pool.data4move_agent(0.9)

print("hello")