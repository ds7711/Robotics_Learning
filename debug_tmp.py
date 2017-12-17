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

final_rewards = np.asarray(trj_pool.final_rewards)
pos_rewards = final_rewards[final_rewards > 0]

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


good_move_idx = move_y >= 0
bad_move_idx = np.logical_not(good_move_idx)
good_move_x, good_move_y = move_x[good_move_idx, :], move_y[good_move_idx]
bad_move_x, bad_move_y = move_x[bad_move_idx, :], move_y[bad_move_idx]
est_good_move_y = policy.mover_q.predict(good_move_x)
est_bad_move_y = policy.mover_q.predict(bad_move_x)
print(np.mean(est_bad_move_y < 0), np.mean(est_good_move_y > 0))

pdb.set_trace()
# fit the model
fitted_release_agent = lnl.training2converge(release_agent, release_x, release_y, batch_size=10000,
                                             epochs=100, verbose=0)
fitted_move_agent = lnl.training2converge(move_agent, move_x, move_y, batch_size=10000, epochs=20, verbose=0)


# find good release examples, # test the prediction accuracy
good_release_idx = release_y >= 0
bad_release_idx = np.logical_not(good_release_idx)
good_release_x, bad_release_x = release_x[good_release_idx, :], release_x[bad_release_idx, :]
good_release_y, bad_release_y = release_y[good_release_idx], release_y[bad_release_idx]
# see predictions from releaser_q
est_good_release_y = policy.releaser_q.predict(good_release_x)
est_bad_release_y = policy.releaser_q.predict(bad_release_x)
print(np.mean(est_bad_release_y < 0), np.mean(est_good_release_y > 0))


idxes = np.argsort(trj_pool.final_rewards)[::-1]
good_move_idx = move_y >= 0
bad_move_idx = np.logical_not(good_move_idx)
good_move_x, good_move_y = move_x[good_move_idx, :], move_y[good_move_idx]
bad_move_x, bad_move_y = move_x[bad_move_idx, :], move_y[bad_move_idx]
est_good_move_y = policy.mover_q.predict(good_move_x)
est_bad_move_y = policy.mover_q.predict(bad_move_x)
print(np.mean(est_bad_move_y < 0), np.mean(est_good_move_y > 0))

pdb.set_trace()

final_rewards = np.asarray(trj_pool.final_rewards)[trj_pool.good_idxes]
avg_reward = np.mean(final_rewards)
print(avg_reward)

new_pool_trj = lnl.TrajectoryPool(max_trajectories=10000, env=env)
new_score_list = []
# pdb.set_trace()

###############################################
policy.mover_q_ub = mover_q_ub
print(mover_q_ub)
###############################################


for iii in range(200):
    ra = copy.deepcopy(env.ini_ra)
    states_list, mas_list, ras_list, rewards_list, score = policy.power_exploring_trajectory(ra, 0.9,
                                                                                             0.5 + avg_reward / 2,
                                                                                             threshold, 0.1)
    new_score_list.append(score)
    new_pool_trj.add_trj(states_list, mas_list, ras_list, rewards_list)
    # print(iii)
    pass

tmp = np.asarray(new_pool_trj.final_rewards)
print(tmp[tmp>0])

pdb.set_trace()


# generate a reachable trajectory
ra = copy.deepcopy(env.ini_ra)
num_time_steps = 25
for _ in range(num_time_steps):
    # action = env.action_combinations[-1]
    action = np.asarray([0, 0, 0, 0, 100, 100])
    ra.update(action, 0)

ra.update(action, 1)
pos, vel = ra.loc_joints()[-1][:-1], ra.cal_ee_speed()[:-1]
print(pos, vel)
ball = knl.BallObj(pos, vel, env)
reward, dist2t = lnl.reward_function(pos, vel, threshold, env.hoop_position, env.gravity)
trj = ball.update()
print(trj[592])