import copy
import time
import numpy as np
from collections import defaultdict
import csv
from agents.ou_noise import OUNoise
from agents.agent import DDPG
from task import Task
import matplotlib.pyplot as plt
import keras
import sys

from plot_functions import plot_results, plot_training_historic


# Run task with agent
def run_test_episode(agent : DDPG, task : Task, file_output):
    print('\nRunning test episode ...')

    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4' ,'reward']
    results = {x : [] for x in labels}

    aux_noise = copy.copy(agent.noise)
    agent.noise = OUNoise(agent.action_size, 0.0, 0.0, 0.0)

    state = agent.reset_episode() # start a new episode
    rewards_lists = defaultdict(list)
    print('state', state)
    print('state.shape', state.shape)

    # Run the simulation, and save the results.
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        while True:
            rotor_speed = agent.act(state)
            rotor_speeds = np.array([rotor_speed ] *4)
            # rotor_speeds = [405]*4
            # rotor_speeds = [500, 490, 500, 500]
            next_state, reward, done, new_rewards = task.step(rotor_speeds)
            for key, value in new_rewards.items():
                rewards_lists[key].append(value)

            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list \
                (rotor_speeds) + [reward]
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)

            state = next_state


            if done:
                break

    # Restore noise
    agent.noise = copy.copy(aux_noise)

    print('Finished test episode!\n')
    return results, rewards_lists


# %% Parameters
class Params:
    def __init__(self):
        self.extra_text = ''
        self.exploration_mu = 0.0
        self.exploration_theta = 0.0
        self.exploration_sigma = 0.0
        self.actor_learning_rate = 0.0
        self.critic_learning_rate = 0.0
        self.tau = 0.0
        self.actor_net_cells = [0]
        self.critic_net_cells = [0]
        self.gamma = 0.0

    def __str__(self):

        txt_nets = '__actor_net'
        for num in self.actor_net_cells:
            txt_nets += '_{}'.format(num)

        txt_nets += '__critic_net'
        for num in self.critic_net_cells:
            txt_nets += '_{}'.format(num)

        param_str = 'mu_{:1.3f}__theta_{:1.3f}__sig_{:1.3f}__alr_{:1.1e}__clr_{:1.1e}__tau_{:1.4f}__gam_{}{}'.format(
            self.exploration_mu,
            self.exploration_theta,
            self.exploration_sigma,
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.tau,
            self.gamma,
            txt_nets
        )

        if self.extra_text != '':
            param_str += '__' + self.extra_text

        return param_str

def run_training(agent, task, params, num_episodes, file_output):
    # %% Training with agent
    num_episodes_to_plot = max(40, num_episodes/5)
    plt.close('all')
    keras.initializers.Initializer()
    results, rewards_lists = run_test_episode(agent, task, file_output)
    plot_results(results,
                 task.target_pos,
                 'Run without training',
                 rewards_lists,
                 num=0,
                 params=params)

    # plt.show();import sys;sys.exit()

    # Train
    max_reward = -np.inf
    last_i_max_reward = 300
    history = {'total_reward': [], 'score': [], 'i_episode': []}
    start = time.time()
    done = False
    i_episode = 1
    stuck_counter = 0
    while i_episode < num_episodes + 1:
        state = agent.reset_episode()  # start a new episode
        t_episode = 0
        time_step_episode = 0
        cum_sum_actions = np.array([0.0])

        start = time.time()
        while True:
            action = agent.act(state, (i_episode - 1) % 100 == 0)
            actions = np.array(action * 4)
            next_state, reward, done, _ = task.step(actions)

            agent.step(actions, reward, next_state, done)

            cum_sum_actions += action
            time_step_episode += 1

            if i_episode % 20 == 0:
                print(
                    "  Ep:{: 4d}. Step:{: 4d}, reward: {:5.1f}, noise(sigma: {:6.4f},"
                    " theta: {:5.3f}, state, {:5.1f})(action:{:6.1f}), "
                    "(state:{:6.1f}, {:5.1f}, {:4.1f}), "
                    "(next_state:{:6.1f}, {:5.1f}, {:4.1f})"
                    " pose ({:6.1f},{:6.1f},{:6.1f},{:6.1f},{:6.1f},{:6.1f}) done: {}".
                        format(i_episode,
                               time_step_episode,
                               reward,
                               agent.noise.sigma,
                               agent.noise.theta,
                               agent.noise.state[0],
                               action[0],
                               *[val for val in state],
                               *[val for val in next_state],
                               *[pose for pose in task.sim.pose],
                               done
                               )
                )

            state = next_state

            t_episode += 0.06  # each step is 3 times 20 ms (50Hz)
            if done:

                episode_time = time.time() - start
                start = time.time()
                i_episode += 1

                # Slowly decrease noise if everything goes all right.
                # agent.noise.sigma = params.exploration_sigma/i_episode
                # agent.noise.theta = params.exploration_theta/i_episode

                if len(history['i_episode']) > 1:
                    history['i_episode'].append(history['i_episode'][-1] + 1)
                else:
                    history['i_episode'].append(1)
                history['total_reward'].append(agent.total_reward)
                history['score'].append(agent.score)

                print(
                    "\rEpisode:{: 4d} (stuck:{: 5d}), score: {:7.1f}, reward: {:8.2f}, noise(sigma: {:6.3f}, theta: {:6.3f}, state, {:6.1f})(action:{:6.1f}). Time: {:5.1f} s.".
                        format(i_episode,
                               stuck_counter,
                               agent.score,
                               agent.total_reward,
                               agent.noise.sigma,
                               agent.noise.theta,
                               *[rotor for rotor in agent.noise.state],
                               *[cum_sum_action / time_step_episode for cum_sum_action in cum_sum_actions],
                               episode_time
                               ),
                    end="")

                break
        sys.stdout.flush()

        if i_episode % num_episodes_to_plot == 0 and stuck_counter == 0:
            results, rewards_lists = run_test_episode(agent, task, file_output)
            plot_results(results,
                         task.target_pos,
                         'Run after training for {} episodes.'.format(i_episode),
                         rewards_lists,
                         num=i_episode,
                         params=params)
        if (max_reward < reward) and (last_i_max_reward + 20 < i_episode):
            results, rewards_lists = run_test_episode(agent, task, file_output)
            plot_results(results,
                         task.target_pos,
                         'New max at: {} episodes.'.format(i_episode),
                         rewards_lists,
                         i_episode,
                         params=params)
            max_reward = reward
            last_i_max_reward = i_episode
            plt.close()

    print('\nTime training: {:.1f} seconds\n'.format(time.time() - start))

    plot_training_historic(history, params)

    results, rewards_lists = run_test_episode(agent, task, file_output)

    plot_results(results,
                 task.target_pos,
                 'Run after training for {} episodes.'.format(num_episodes),
                 rewards_lists,
                 num=0,
                 params=params)

    plt.show(block=False)
