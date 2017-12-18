import numpy as np
import pdb
import kinematics_library as knl
import learning_library as lnl
import environment
# import matplotlib.pylab as plt
import copy
import IPython


# debug training
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

pdb.set_trace()




# initialize objects
env = environment.Env2D()
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

# find one good trj and one bad trj to compare the probability of choosing that before and after training
final_rewards = np.asarray(trj_pool.final_rewards)
pos_rewards = final_rewards[final_rewards > 0]
idxes = np.argsort(final_rewards)[::-1]
good_idx = idxes[0]
bad_idx = idxes[-1]

tmp_idx = good_idx
move_actions = trj_pool.move_actions_list[tmp_idx]
release_actions = trj_pool.release_actions_list[tmp_idx]
state_list = trj_pool.states_list[tmp_idx]
ra = copy.deepcopy(env.ini_ra)

print("debug start!")
# pdb.set_trace()
# _, release_q_vals = policy._test_release_q(ra, states_list, threshold)
state_list, move_q_vals, release_q_vals, rewards, score = policy.test_move_q(ra, move_actions, release_actions, threshold)
release_x, release_y = trj_pool.data4release_agent()
move_x, move_y = trj_pool.data4move_agent(0.9)

# fit the model
fitted_release_agent = lnl.training2converge(release_agent, release_x, release_y, batch_size=10000,
                                             epochs=100, verbose=0)
fitted_move_agent = lnl.training2converge(move_agent, move_x, move_y, batch_size=10000, epochs=20, verbose=0)

# test after training
idxes = np.argsort(final_rewards)[::-1]
good_idx = idxes[0]
bad_idx = idxes[-1]

tmp_idx = good_idx
move_actions = trj_pool.move_actions_list[tmp_idx]
release_actions = trj_pool.release_actions_list[tmp_idx]
state_list = trj_pool.states_list[tmp_idx]
ra = copy.deepcopy(env.ini_ra)

print("debug start!")
# _, release_q_vals = policy._test_release_q(ra, state_list, threshold)
state_list, move_q_vals, release_q_vals, rewards, score = policy.test_move_q(ra, move_actions, release_actions, threshold)


# caculate the q_value criterion for releasing the ball
good_idx = release_y > 0
good_release_x = release_x[good_idx, :]
ra = copy.deepcopy(env.ini_ra)
q_vals = np.squeeze(policy.releaser_q.predict(good_release_x))

criterion = 3
q_val_threshold = (np.mean(q_vals) + criterion * np.max(q_vals)) / (1.0 + criterion)


new_pool_trj = lnl.TrajectoryPool(max_trajectories=10000, env=env)
new_score_list = []

for _ in range(200):
    ra = copy.deepcopy(env.ini_ra)
    states_list, mas_list, ras_list, rewards_list, score = policy.power_exploring_trajectory(ra, 0.9,
                                                                                             q_val_threshold,
                                                                                             threshold, 0.2)
    new_score_list.append(score)
    new_pool_trj.add_trj(states_list, mas_list, ras_list, rewards_list)
    states_list, mas_list, ras_list, rewards_list, score = policy.power_exploring_trajectory(ra, 0.9,
                                                                                             q_val_threshold,
                                                                                             threshold, 0.2)


tmp = np.asarray(new_pool_trj.final_rewards)
print(tmp[tmp > 0])

pdb.set_trace()


# generate a reachable trajectory
ra = copy.deepcopy(env.ini_ra)
num_time_steps = 25
for iii in range(num_time_steps):
    # action = env.action_combinations[-1]
    action = np.asarray([0, 0, 0, 0, 100, 100])
    ra.update(action, 0)

ra.update(action, 1)
pos, vel = ra.loc_joints()[-1][:-1], ra.cal_ee_speed()[:-1]
print(pos, vel)
ball = knl.BallObj(pos, vel, env)
reward, dist2t = lnl.reward_function(pos, vel, threshold, env.hoop_position, env.gravity)
trj = ball.update()
print(trj[590])