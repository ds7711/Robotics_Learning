import numpy as np
import kinematics_library as knl
import environment
import matplotlib.pylab as plt
import copy


# 1. debug for Ball Object
pos = np.asarray([3, 3, 0.5])
vel = np.asarray([7, 0, 6])
env = environment.Env2D()
env.hoop_position = np.asarray([7.5, 3, 2])

tmp_ball = knl.BallObj(pos, vel, env)

ball_trajectory = tmp_ball.update()

plt.scatter(ball_trajectory[:, 0], ball_trajectory[:, 2])
plt.scatter(env.hoop_position[0], env.hoop_position[2], marker="*")

print(knl.ball2hoop(pos, vel, env.hoop_position, env.gravity))