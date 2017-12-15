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
