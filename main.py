"""
1. reward_function: continuous, not very discrete, may not be a good idea???
2. Seperate reward function for the releaser and mover??? Use LQR for mover as a planning problem
3. _Epislon_greedy release action has to be changed
4. Trajectory2Data to be completed
5. DataPool to be completed
"""



import numpy as np
import kinematics_library as knl
import environment
import learning_library as lnl
import pdb

# initialize objects
env = environment.Env2D()

release_agent_str = [4, 100, 50, 25, 1]
move_agent_str = [6, 650, 250, 125, 1]

avg_scores, reward_freqs, reward_thres_list = lnl.shaping_training(move_agent_str, release_agent_str, env)

pdb.set_trace()

print("hello123!")





