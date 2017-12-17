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
release_agent = lnl.get_release_agent([4, 30, 10, 1])
move_agent = lnl.get_move_agent([6, 50, 20, 1])
policy = lnl.PolicyObject(move_agent, release_agent, env)


# test the exploration method
threshold = env.ee2hoop - 0.5
trj_pool = lnl.TrajectoryPool(max_trajectories=100000, env=env)
score_list = []
# pdb.set_trace()
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

# find actions that should give positive reward
move_actions_list_bk = np.asarray(trj_pool.move_actions_list)
release_actions_list_bk = np.asarray(trj_pool.release_actions_list)
# idx_list = []
# for iii in range(len(move_actions_list_bk)):
#     tmp_actions = move_actions_list_bk[iii][:-1]
#     if np.sum(tmp_actions[:, -2] > 10) > 2 and np.sum(tmp_actions[:, -1] > 10) > 2:
#         idx_list.append(iii)
idx = np.argmax(final_rewards)
move_actions = move_actions_list_bk[idx]
release_actions = release_actions_list_bk[idx]
ra = copy.deepcopy(env.ini_ra)


a, b, c, d = policy.test_move_q(ra, move_actions, release_actions, threshold)

release_x, release_y = trj_pool.data4release_agent()
move_x, move_y = trj_pool.data4move_agent(0.9)

a, b = release_agent.layers[0].get_weights()
a, b = np.copy(a), np.copy(b)

ra_hist = release_agent.fit(release_x, release_y[:, np.newaxis], batch_size=10000, epochs=500, validation_split=0.2)
ma_hist = move_agent.fit(move_x, move_y[:, np.newaxis], batch_size=10000, epochs=100, validation_split=0.2)

c, d = policy.releaser_q.layers[0].get_weights()

tmp = pos_rewards
mover_q_ub = np.mean(tmp)
releaser_q_avg = np.mean(tmp)



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