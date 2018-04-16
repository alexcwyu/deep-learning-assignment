import sys

import numpy as np
from agents.agent import DDPG
from task import Task
from utils import *

import keras

keras.initializers.Initializer()
runtime = 5.
num_episodes = 2000
target_pos = np.array([0., 0., 10.])  #vertical up
init_pose = np.array([0., 0., 1., 0., 0., 0.])
init_velocities = np.array([0., 0., 0.])
init_angle_velocities = np.array([0., 0., 0.])

# parameter
actor_batch_normalized = True
actor_dropout = True
actor_dropout_rate = 0.8
actor_lr = 0.0001
actor_beta1 = 0.9

critic_batch_normalized = True
critic_dropout = True
critic_dropout_rate = 0.8
critic_lr = 0.001
critic_beta1 = 0.9

exploration_mu = 0
exploration_theta = 0.15
exploration_sigma = 0.2

buffer_size = 100000
batch_size = 64

gamma = 0.99
tau = 0.01

task = Task(init_pose=init_pose,
            init_velocities=init_velocities,
            init_angle_velocities=init_angle_velocities,
            runtime=runtime,
            target_pos=target_pos)
agent = DDPG(task,
             actor_batch_normalized=actor_batch_normalized,
             actor_dropout=actor_dropout,
             actor_dropout_rate=actor_dropout_rate,
             actor_lr=actor_lr,
             actor_beta1=actor_beta1,

             critic_batch_normalized=critic_batch_normalized,
             critic_dropout=critic_dropout,
             critic_dropout_rate=critic_dropout_rate,
             critic_lr=critic_lr,
             critic_beta1=critic_beta1,

             exploration_mu=exploration_mu,
             exploration_theta=exploration_theta,
             exploration_sigma=exploration_sigma,

             buffer_size=buffer_size,
             batch_size=batch_size,

             gamma=gamma,
             tau=tau)

all_results = {x: [] for x in result_labels}
best_results = None
best_reward = None
best_iter = None

for i_episode in range(1, num_episodes + 1):
    state = agent.reset_episode()  # start a new episode
    curr_iter_results = {x: [] for x in result_labels}

    count = 0
    while True:
        count+=1
        action = agent.act(state)
        next_state, reward, done, rewards = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state

        result = create_result(i_episode, task, action, reward, done, rewards)
        merge_result(all_results, result)
        merge_result(curr_iter_results, result)

        if done:
            if best_reward == None or reward > best_reward:
                best_results = curr_iter_results
                best_reward = reward
                best_iter = i_episode
            print("iter: {}, reward: {:.2f}, count: {}, x: {:.2f}, y: {:.2f}, z: {:.2f}, r1: {:.2f}, r2: {:.2f}, r3: {:.2f}, r4: {:.2f}".format(i_episode, reward, count, task.sim.pose[0], task.sim.pose[1], task.sim.pose[2], action[0], action[1], action[2], action[3]))
            #if i_episode >= (num_episodes - 10) or i_episode % 20 ==0:
            #    print("iter: {}, reward: {:.2f}, count: {}, x: {:.2f}, y: {:.2f}, z: {:.2f}".format(i_episode, reward, count, task.sim.pose[0], task.sim.pose[1], task.sim.pose[2]))
            break



print(best_results)