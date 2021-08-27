#!/usr/bin/env python3

import random
import numpy as np
import tensorflow as tf

from utils import capture, render_gif

class DQNAgent:
    def __init__(self, config, sandbox, env, action_space):
        self.SANDBOX = sandbox
        self.ENV = env
        self.ACTION_SPACE = action_space

        self.EPSILON = config['epsilon']
        self.EPSILON_RANDOM_FRAMES = config['epsilon_random_frames']
        self.EPSILON_GREEDY_FRAMES = config['epsilon_greedy_frames']
        self.EPSILON_MIN = config['epsilon_min']
        self.EPSILON_MAX = config['epsilon_max']
        self.EPSILON_ANNEALER = (config['epsilon_max'] - config['epsilon_min'])

        self.BATCH_SIZE = config['batch_size']
        self.GAMMA = config['gamma']
        self.UPDATE_TARGET_NETWORK = config['update_target_network']
        self.TAU = config['tau']

        self.DOUBLE = config['double']
        self.USE_PER = config['use_per']

    def get_action(self, state, model):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()
        return action

    def exploration(self, frame_count, state, model):
        if frame_count < self.EPSILON_RANDOM_FRAMES or self.EPSILON > np.random.rand(1)[0]:
            action_idx = random.randrange(self.ACTION_SPACE)
        else:
            action_idx = self.get_action(state, model)

        self.EPSILON -= self.EPSILON_ANNEALER / self.EPSILON_GREEDY_FRAMES
        self.EPSILON = max(self.EPSILON, self.EPSILON_MIN)
        return action_idx

    def learn(self, memory, model, model_target, optimizer):
        # Sample from replay buffer
        if self.USE_PER:
            samples, indices, priorities = memory.sample(self.BATCH_SIZE)
            action_sample, state_sample, state_next_sample, reward_sample, terminal_sample = samples
        else:
            action_sample, state_sample, state_next_sample, reward_sample, terminal_sample = memory.sample()

        # Double Q-Learning, decoupling selection and evaluation of the action seletion with the current DQN model.
        q = model.predict(state_next_sample)
        target_q = model_target.predict(state_next_sample)

        # Build the updated Q-values for the sampled future states - DQN / DDQN
        if self.DOUBLE:
            max_q = tf.argmax(q, axis=1)
            max_actions = tf.one_hot(max_q, self.ACTION_SPACE)
            q_samp = reward_sample + self.GAMMA * tf.reduce_sum(tf.multiply(target_q, max_actions), axis=1)
        else:
            q_samp = reward_sample + self.GAMMA * tf.reduce_max(target_q, axis=1)      # Bellman Equation

        q_samp = q_samp * (1 - terminal_sample) - terminal_sample                       # If final frame set the last value to -1
        masks = tf.one_hot(action_sample, self.ACTION_SPACE)                            # Create a mask so we only calculate loss on the updated Q-values

        with tf.GradientTape() as tape:
            q_values = model(state_sample)                                              # Train the model on the states and updated Q-values
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)              # Apply the masks to the Q-values to get the Q-value for action taken
            loss = tf.keras.losses.Huber()(q_samp, q_action)                            # Calculate loss between new Q-value and old Q-value

        # Backpropagation
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if self.USE_PER:
            td_error = abs(q_action - q_samp) + 0.1
            memory.update(indices, td_error)

        return float(loss)

    def fixed_target(self, frame_count, model, model_target):
        if frame_count % self.UPDATE_TARGET_NETWORK == 0:
            model_target.set_weights(model.get_weights())

    def soft_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.TAU + a * (1 - self.TAU))

    def td_error(self, model, model_target, action, state, state_next, reward, terminal):
        state = tf.expand_dims(state, 0)
        state_next = tf.expand_dims(state_next, 0)

        q = model.predict(state_next)[0]
        target_q = model_target.predict(state_next)

        if self.DOUBLE:
            max_q = np.argmax(q)
            max_actions = tf.one_hot(max_q, self.ACTION_SPACE)
            q_samp = reward + self.GAMMA * tf.reduce_sum(tf.multiply(target_q, max_actions), axis=1)
        else:
            q_samp = reward + self.GAMMA * tf.reduce_max(target_q, axis=1)      # Bellman Equation

        q_samp = q_samp * (1 - terminal) - terminal
        masks = tf.one_hot(action, self.ACTION_SPACE)

        q_values = model(state)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

        td_error = abs(q_action - q_samp) + 0.1
        return td_error

    def evaluate(self, model, log_dir, episode_id):
        info, prev_info, stack, state = self.SANDBOX.reset(self.ENV)

        frames = []
        episode_reward = 0

        while not self.ENV.is_episode_finished():
            frames = capture(self.ENV, self.SANDBOX, frames)                                                                        # Capture gameplay experience
            action = self.get_action(state, model)                                                                                  # Predict action Q-values from environment state and take best action
            state_next, reward, terminal, info = self.SANDBOX.step(self.ENV, stack, prev_info, action, self.ACTION_SPACE)           # Apply the sampled action in our environment

            episode_reward += reward

            prev_info = info
            state = state_next

            if terminal:
                break

        render_gif(frames, log_dir + "/loop_" + str(episode_id) + "_" + str(episode_reward))
        return episode_reward

class PolicyAgent:
    def __init__(self, config, sandbox, env, action_space):
        self.SANDBOX = sandbox
        self.ENV = env
        self.ACTION_SPACE = action_space

        self.GAMMA = config['gamma']
        self.STATE_SIZE = (config['window_length'], config['input_shape'][0], config['input_shape'][1])

        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def act(self, state, model):
        policy = model.predict(state)[0]
        action = np.random.choice(self.ACTION_SPACE, p=policy)
        return action

    def push(self, state, action_idx, reward):
        action = np.zeros([self.ACTION_SPACE])
        action[action_idx] = 1

        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)

    def discount_rewards(self):
        sum_reward = 0
        discounted_r = np.zeros_like(self.reward_history)
        for i in reversed(range(0,len(self.reward_history))):
            if self.reward_history[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                sum_reward = 0
            sum_reward = sum_reward * self.GAMMA + self.reward_history[i]
            discounted_r[i] = sum_reward

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r

    def learn(self, model):
        states = np.vstack(self.state_history)                                                      # reshape memory to appropriate shape for training
        actions = np.vstack(self.action_history)

        discounted_r = self.discount_rewards()

        model.fit(states, actions, sample_weight=discounted_r, epochs=1, verbose=0)                 # training PG network
        self.state_history, self.action_history, self.reward_history = [], [], []                   # Reset training memory
