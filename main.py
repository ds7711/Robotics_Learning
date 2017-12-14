"""
1. reward_function: continuous, not very discrete, may not be a good idea???
2. Seperate reward function for the releaser and mover??? Use LQR for mover as a planning problem
3. Power exploring trajectory to be changed
"""



import numpy as np
import kinematics_library as knl
import environment
import learning_library as lnl

# import the environment
env = environment.Env2D()
ini_ra = env.ini_ra  # initial configuration of the robot

# get the intial release and move agent
# joint angles, angular velocities, q_value of release as output
releaser_q = lnl.get_release_agent([4, 20, 5, 1])
# joint angles, velcoties, and accelerations as input, q_value as output
mover_q = lnl.get_move_agent([6, 50, 25, 1])  # use LQR planner? se




