import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.axes3d import Axes3D
from collections import defaultdict

import numpy as np

result_labels = ['iter', 'time',
                 'x', 'y', 'z',
                 'phi', 'theta', 'psi',
                 'x_velocity', 'y_velocity', 'z_velocity',
                 'phi_velocity', 'theta_velocity', 'psi_velocity',
                 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4',
                 'done', 'reward', 'surviving_reward', 'distance_reward', 'speed_reward', 'angular_speed_reward', 'similar_rotors_reward']


def create_result(iter, task, action, reward, done, rewards = None):
    rewards = rewards if rewards else defaultdict(float)
    return [iter, task.sim.time] + \
           list(task.sim.pose) + \
           list(task.sim.v) + \
           list(task.sim.angular_v) + \
           list(action) + \
           [done, reward,
            rewards.get('surviving', 0),
            rewards.get('distance', 0),
            rewards.get('speed', 0),
            rewards.get('angles', 0),
            rewards.get('angular_speed', 0),
            rewards.get('similar_rotors', 0)]



def merge_result(results, latest_result):
    for ii in range(len(result_labels)):
        results[result_labels[ii]].append(latest_result[ii])



def plot_point3d(ax, x, y, z, **kwargs):
    ax.scatter([x], [y], [z], **kwargs)
    ax.text(x, y, z, "({:.1f}, {:.1f}, {:.1f})".format(x, y, z))


def plot_flight_path(results, target=None):
    path = [[results['x'][i], results['y'][i], results['z'][i]] for i in range(len(results['x']))]
    path = np.array(path)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.plot3D(path[:, 0], path[:, 1], path[:, 2], 'gray')

    if target is not None:
        plot_point3d(ax, *target[0:3], c='y', marker='X', s=100, label='target')

    plot_point3d(ax, *path[0, 0:3], c='g', marker='o', s=50, label='start')
    plot_point3d(ax, *path[-1, 0:3], c='r', marker='o', s=50, label='end')

    ax.legend()


def plot_position(results):
    plt.figure(figsize=(5, 5))
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.title("position vs time")
    plt.ylabel("position")
    plt.legend()


def plot_velocity(results):
    plt.figure(figsize=(5, 5))
    plt.plot(results['time'], results['x_velocity'], label='x_hat')
    plt.plot(results['time'], results['y_velocity'], label='y_hat')
    plt.plot(results['time'], results['z_velocity'], label='z_hat')
    plt.title("velocity vs time")
    plt.ylabel("velocity")
    plt.legend()


def plot_euler_angles(results):
    plt.figure(figsize=(5, 5))
    plt.plot(results['time'], results['phi'], label='phi')
    plt.plot(results['time'], results['theta'], label='theta')
    plt.plot(results['time'], results['psi'], label='psi')
    plt.title("Euler angles vs time")
    plt.ylabel("Euler angles")
    plt.legend()


def plot_euler_angles_velocity(results):
    plt.figure(figsize=(5, 5))
    plt.plot(results['time'], results['phi_velocity'], label='phi_velocity')
    plt.plot(results['time'], results['theta_velocity'], label='theta_velocity')
    plt.plot(results['time'], results['psi_velocity'], label='psi_velocity')
    plt.title("Euler angles velocity vs time")
    plt.ylabel("Euler angles velocity")
    plt.legend()


def plot_revolutions(results):
    plt.figure(figsize=(5, 5))
    plt.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
    plt.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
    plt.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
    plt.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
    plt.title("Rotor revolutions vs time")
    plt.ylabel("revolutions / second")
    plt.legend()


def visualize_result(results):
    plot_position(results)
    plot_velocity(results)
    plot_euler_angles(results)
    plot_euler_angles_velocity(results)
    plot_revolutions(results)
    plot_flight_path(results)

    _ = plt.ylim()
