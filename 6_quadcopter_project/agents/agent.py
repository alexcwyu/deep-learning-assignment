import numpy as np

from .actor import Actor
from .critic import Critic
from .ou_noise import OUNoise
from .replay_buffer import ReplayBuffer


class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""

    def __init__(self, task,
                 actor_batch_normalized=True,
                 actor_dropout=True,
                 actor_dropout_rate=0.8,
                 actor_lr=0.0001,
                 actor_beta1=0.9,
                 critic_batch_normalized=True,
                 critic_dropout=True,
                 critic_dropout_rate=0.8,
                 critic_lr=0.001,
                 critic_beta1=0.9,
                 exploration_mu=0,
                 exploration_theta=0.15,
                 exploration_sigma=0.2,
                 buffer_size=100000,
                 batch_size=64,
                 gamma=0.99,
                 tau=0.01):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high,
                                 actor_batch_normalized, actor_dropout, actor_dropout_rate, actor_lr, actor_beta1)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high,
                                  actor_batch_normalized, actor_dropout, actor_dropout_rate, actor_lr, actor_beta1)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size,
                                   critic_batch_normalized, critic_dropout, critic_dropout_rate, critic_lr, critic_beta1)
        self.critic_target = Critic(self.state_size, self.action_size,
                                    critic_batch_normalized, critic_dropout, critic_dropout_rate, critic_lr, critic_beta1)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = exploration_mu
        self.exploration_theta = exploration_theta
        self.exploration_sigma = exploration_sigma
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # for soft update of target parameters

        # store the reward
        self.total_reward = 0
        self.count = 0

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.total_reward = 0
        self.count = 0
        return state

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

        self.total_reward += reward
        self.count += 1

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        # state = np.reshape(state, [-1, self.state_size])
        # noise = self.noise.sample()#*0.5*(self.action_high - self.action_low)
        # action = np.clip(state + noise, self.action_low, self.action_high)
        # return action

        state = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        # add some noise for exploration
        noise = self.noise.sample()
        action = np.clip(action + noise, self.action_low, self.action_high)
        return action

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1,
                                                                                                        self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]),
                                      (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
