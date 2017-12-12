import numpy as np
import kinematics_library as knl
import learning_library as lnl
import environment
import matplotlib.pylab as plt
import copy
import pdb


# 4. test the random_explorer
env = environment.Env2D()
ra = copy.deepcopy(env.ini_ra)
threshold = 10.0
pdb.set_trace()
states, move_actions, release_actions, rewards = lnl.random_explorer(ra, 3, threshold, env)

print(states)

# 3. test reward function
env = environment.Env2D()
ra = copy.deepcopy(env.ini_ra)

threshold = 1.0

ra.joint_angles = np.asarray([0.0, 0, 0, 0, np.pi/4, 0])
ra.angular_velocities = np.asarray([0.0, 0, 0, 0, 3.5 - 2, 0])
ee_speed = ra.cal_ee_speed()
ee_pos = ra.loc_joints()[-1]

tmp_ball = knl.BallObj(ee_pos, ee_speed, env)
ball_trj = tmp_ball.update()

rotation_point = ra.loc_joints()[-3]
plt.scatter(rotation_point[0], rotation_point[2], marker="o")

plt.scatter(ball_trj[:, 0], ball_trj[:, 2])

plt.scatter(env.hoop_position[0], env.hoop_position[2], marker="*")

reward = lnl.reward_function(ee_pos, ee_speed, threshold, env.hoop_position, env.gravity)
dist2t = knl.ball2hoop(ee_pos, ee_speed, env.hoop_position, env.gravity)
print(reward, dist2t)

reward_list = []
dist2t_list = []

threshold_list = np.arange(0.85, 5, 0.5)
for threshold in threshold_list:
    # threshold = 1
    reward_list = []
    dist2t_list = []
    for delta_speed in np.arange(0, 10, 1e-1):
        ra.joint_angles = np.asarray([0.0, 0, 0, 0, np.pi / 4, 0])
        ra.angular_velocities = np.asarray([0.0, 0, 0, 0, 0 + delta_speed, 0])
        ee_speed = ra.cal_ee_speed()
        ee_pos = ra.loc_joints()[-1]
        reward = lnl.reward_function(ee_pos, ee_speed, threshold, env.hoop_position, env.gravity)
        dist2t = knl.ball2hoop(ee_pos, ee_speed, env.hoop_position, env.gravity)
        reward_list.append(reward)
        dist2t_list.append(dist2t)
    plt.scatter(dist2t_list, reward_list)


# 2. debug for robot arm
env = environment.Env2D()
ra = copy.deepcopy(env.ini_ra)
print(ra.initial_joint_angles, ra.joint_angles)
ra.initial_joint_angles = 0
print(ra.initial_joint_angles, ra.joint_angles)
print(env.ini_ra.joint_angles)


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