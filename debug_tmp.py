import numpy as np
import pdb
import kinematics_library as knl
import learning_library as lnl
import environment
import matplotlib.pylab as plt
import copy


# debug the epislon greedy policy
env = environment.Env2D()
ra = copy.deepcopy(env.ini_ra)

# initialize the mover and releaser agent
mover = lnl.get_move_agent([6, 20, 1])
releaser = lnl.get_release_agent([4, 10, 1])

policy = lnl.PolicyObject(mover, releaser, env)

states, move_actions, release_actions, rewards = policy.epsilon_greedy_trajectory(ra, 0.9, 0.1, 1)

ra = copy.deepcopy(env.ini_ra)
states1, move_actions1, release_actions1, rewards1 = policy.power_exploring_trajectory(ra, 0.9, 0.1, 1, 0.1)

print 1 + 1


