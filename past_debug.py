import numpy as np
import kinematics_library as knl
import learning_library as lnl
import environment
import matplotlib.pylab as plt
import copy
import pdb


# 7. test the training results

env = environment.Env2D()
ra = copy.deepcopy(env.ini_ra)
release_agent = lnl.get_release_agent([4, 100, 50, 25, 1])
move_agent = lnl.get_move_agent([6, 650, 250, 125, 1])
policy = lnl.PolicyObject(move_agent, release_agent, env)

# test the exploration method
threshold = env.ee2hoop - 0.5
trj_pool = lnl.TrajectoryPool(max_trajectories=100000, env=env)
score_list = []

print(env.action_combinations)
for iii in range(10000):
    ra = copy.deepcopy(env.ini_ra)
    states_list, mas_list, ras_list, rewards_list, score = policy.random_explorer(ra, np.random.randint(1, 30), threshold, noise=0.0)
    # if rewards_list[-1] > 0.8:
    #     pdb.set_trace()
    score_list.append(score)
    trj_pool.add_trj(states_list, mas_list, ras_list, rewards_list)

# get good & bad examples and see the predictions from fitted model
release_x, release_y = trj_pool.data4release_agent()
move_x, move_y = trj_pool.data4move_agent(0.9)

# test predictions before training
good_release_idx = release_y >= 0
bad_release_idx = np.logical_not(good_release_idx)
good_release_x, bad_release_x = release_x[good_release_idx, :], release_x[bad_release_idx, :]
good_release_y, bad_release_y = release_y[good_release_idx], release_y[bad_release_idx]
# see predictions from releaser_q
est_good_release_y = policy.releaser_q.predict(good_release_x)
est_bad_release_y = policy.releaser_q.predict(bad_release_x)
print(np.mean(est_bad_release_y < 0), np.mean(est_good_release_y > 0))

fitted_release_agent = lnl.training2converge(release_agent, release_x, release_y, batch_size=10000,
                                             epochs=100, verbose=0)
fitted_move_agent = lnl.training2converge(move_agent, move_x, move_y, batch_size=10000, epochs=20, verbose=0)

good_release_idx = release_y >= 0
bad_release_idx = np.logical_not(good_release_idx)
good_release_x, bad_release_x = release_x[good_release_idx, :], release_x[bad_release_idx, :]
good_release_y, bad_release_y = release_y[good_release_idx], release_y[bad_release_idx]
# see predictions from releaser_q
est_good_release_y = policy.releaser_q.predict(good_release_x)
est_bad_release_y = policy.releaser_q.predict(bad_release_x)
print(np.mean(est_bad_release_y < 0), np.mean(est_good_release_y > 0))


good_move_idx = move_y >= 0
bad_move_idx = np.logical_not(good_move_idx)
good_move_x, good_move_y = move_x[good_move_idx, :], move_y[good_move_idx]
bad_move_x, bad_move_y = move_x[bad_move_idx, :], move_y[bad_move_idx]
est_good_move_y = policy.mover_q.predict(good_move_x)
est_bad_move_y = policy.mover_q.predict(bad_move_x)
print(np.mean(est_bad_move_y < 0), np.mean(est_good_move_y > 0))


# 6. debug _propagate_rewards in TrajectoryPool
rewards = np.zeros(10)
rewards[-1] = -0.9

new_rewards = lnl.TrajectoryPool._propogate_rewards(rewards, 0.9)


# 5. debug for the policy object
# debug the epislon greedy policy
env = environment.Env2D()
ra = copy.deepcopy(env.ini_ra)

# initialize the mover and releaser agent
mover = lnl.get_move_agent([6, 20, 1])
releaser = lnl.get_release_agent([4, 10, 1])
threshold = 5
policy = lnl.PolicyObject(mover, releaser, env)


ra = copy.deepcopy(env.ini_ra)
print(ra.state)
states2, move_actions2, release_actions2, rewards2 = policy.greedy_plus_random_explorer(ra, 0.9, 0.9, threshold)

ra = copy.deepcopy(env.ini_ra)
print(ra.state)
states3, move_actions3, release_actions3, rewards3 = policy.power_plus_random_explorer(ra, 0.9, 0.9, threshold, 0.1)

ra = copy.deepcopy(env.ini_ra)
print(ra.state)
states, move_actions, release_actions, rewards = policy.random_explorer(ra, 20, threshold)

ra = copy.deepcopy(env.ini_ra)
print(ra.state)
states1, move_actions1, release_actions1, rewards1 = policy.power_exploring_trajectory(ra, 0.9, 0.9, threshold, 0.1)

ra = copy.deepcopy(env.ini_ra)
print(ra.state)
states, move_actions, release_actions, rewards = policy.epsilon_greedy_trajectory(ra, 0.9, 0.9, threshold)

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