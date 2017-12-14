import numpy as np
import pdb
import kinematics_library as knl
import learning_library as lnl
import environment
# import matplotlib.pylab as plt
import copy
import IPython


# debug the epislon greedy policy
env = environment.Env2D()
ra = copy.deepcopy(env.ini_ra)

# initialize the mover and releaser agent
mover = lnl.get_move_agent([6, 20, 1])
releaser = lnl.get_release_agent([4, 10, 1])
threshold = 5
policy = lnl.PolicyObject(mover, releaser, env)


# pdb.set_trace()
ra = copy.deepcopy(env.ini_ra)
print(ra.state)
states, move_actions, release_actions, rewards = policy.epsilon_greedy_trajectory(ra, 0.9, 0.1, threshold)

ra = copy.deepcopy(env.ini_ra)
print(ra.state)
states2, move_actions2, release_actions2, rewards2 = policy.greedy_plus_random_explorer(ra, 0.9, 0.1, threshold)

ra = copy.deepcopy(env.ini_ra)
print(ra.state)
states3, move_actions3, release_actions3, rewards3 = policy.power_plus_random_explorer(ra, 0.9, 0.1, threshold, 0.1)

print("hello")


ra = copy.deepcopy(env.ini_ra)
print(ra.state)
states, move_actions, release_actions, rewards = policy.random_explorer(ra, 20, threshold)

ra = copy.deepcopy(env.ini_ra)
print(ra.state)
states1, move_actions1, release_actions1, rewards1 = policy.power_exploring_trajectory(ra, 0.9, 0.1, threshold, 0.1)




