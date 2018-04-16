
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os


def save_figure(plt, file_name, params):

    folder_name = str(params)

    try:
        figure_path = os.path.join(os.getcwd(),
                                   'figures',
                                   folder_name,
                                   file_name)
        verify_folder_for_file(figure_path)
        plt.savefig(figure_path)
    except:
        print('Image open or something, could not save.')


def verify_folder_for_file(file_path):

    folder = os.path.dirname(file_path)
    if not os.path.isdir(folder):
        os.makedirs(folder)


def plot_results(results, target_pos, title, rewards_lists, num, params):
    # %matplotlib inline

    if results['time'][-1] < 0.1:
        print('Episode too short, skipping plot')
        return

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(3, 2, 1)
    ax3 = fig.add_subplot(3, 2, 3)
    ax5 = fig.add_subplot(3, 2, 5)

    ax2 = fig.add_subplot(3, 2, 2)
    ax4 = fig.add_subplot(3, 2, 4)
    ax6 = fig.add_subplot(3, 2, 6, projection='3d')

    ax1.plot(results['time'], results['x'], label='x')
    ax1.plot(results['time'], results['y'], label='y')
    ax1.plot(results['time'], results['z'], label='z')
    ax1.legend()
    ax1.grid()

    ax3.plot(results['time'], results['phi'], label='phi')
    ax3.plot(results['time'], results['theta'], label='theta')
    ax3.plot(results['time'], results['psi'], label='psi')
    ax3.legend()
    ax3.grid()

    ax5.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
    ax5.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
    ax5.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
    ax5.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
    ax5.legend()
    ax5.grid()

    for name, reward in rewards_lists.items():
        ax2.plot(results['time'], reward, label=name)
    ax2.legend()
    ax2.grid()

    ax4.plot(results['time'], results['reward'], label='reward')
    ax4.legend()
    ax4.grid()

    ax6.plot(results['x'], results['y'], results['z'], label='parametric curve')
    ax6.plot([target_pos[0]], [target_pos[1]], [target_pos[2]], 'ro', markersize=12, label='target')
    ax6.plot([results['x'][0]], [results['y'][0]], [results['z'][0]], 'gx', markersize=6, label='start')
    ax6.plot([results['x'][-1]], [results['y'][-1]], [results['z'][-1]], 'bx', markersize=6, label='end')
    ax6.legend()

    ax6.set_xlabel('X axis')
    ax6.set_ylabel('Y axis')
    ax6.set_zlabel('Z axis')

    fig.suptitle(str(params))
    plt.suptitle(title)

    save_figure(plt, 'last_fig_plot_results' + str(num) + '.png', params)


def plot_training_historic(history, params):

    f, (ax1) = plt.subplots(1, 1, figsize=(12, 6))

    N = 20
    total_reward = np.array(history['total_reward'])
    total_reward = np.convolve(total_reward, np.ones((N,)) / N, mode='valid')
    total_reward = np.concatenate([np.array([total_reward[0]]*(N-1)), total_reward])

    ax1.plot(history['i_episode'], total_reward, label='total_reward_filt')
    ax1.set_ylim([min(total_reward), max(total_reward)])
    ax1.legend()
    ax1.grid()

    save_figure(plt, 'results_reward.png', params)

    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 6))

    N = 20
    score = np.array(history['score'])
    score = np.convolve(score, np.ones((N,)) / N, mode='valid')
    score = np.concatenate([np.array([score[0]]*(N-1)), score])

    ax1.plot(history['i_episode'], score, label='score')
    ax1.set_ylim([min(score), max(score)])
    ax1.legend()
    ax1.grid()

    fig.suptitle(str(params))

    save_figure(plt, 'results_score.png', params)
