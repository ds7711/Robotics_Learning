import numpy as np
import kinematics_library as knl
import environment
import matplotlib.pylab as plt
import copy


# debug for the RobotArm
env = environment.Env2D()
ra = copy.deepcopy(env.ini_ra)
tmp = env.action_spaces
tmp = env.ext_action_cmbs
