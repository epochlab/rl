#!/usr/bin/env python3

import random
import numpy as np
import tensorflow as tf

from utils import capture, render_gif

class Agent:
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
        self.UPDATE_AFTER_ACTIONS = config['update_after_actions']
        self.GAMMA = config['gamma']
        self.UPDATE_TARGET_NETWORK = config['update_target_network']
        self.TAU = config['tau']

        self.DOUBLE = config['double']
        self.USE_PER = config['use_per']

    def exploration(self, frame_count, state, model):
        if frame_count < self.EPSILON_RANDOM_FRAMES or self.EPSILON > np.random.rand(1)[0]:
            action_idx = random.randrange(self.ACTION_SPACE)
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action_idx = tf.argmax(action_probs[0]).numpy()

        self.EPSILON -= self.EPSILON_ANNEALER / self.EPSILON_GREEDY_FRAMES
        self.EPSILON = max(self.EPSILON, self.EPSILON_MIN)
        return action_idx

    def learn(self, frame_count, memory, model, model_target, optimizer):
        if frame_count % self.UPDATE_AFTER_ACTIONS == 0 and frame_count > self.BATCH_SIZE:
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

    def static_target(self, frame_count, model, model_target):
        if frame_count % self.UPDATE_TARGET_NETWORK == 0:
            model_target.set_weights(model.get_weights())

    def fixed_q(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.TAU + a * (1 - self.TAU))

    def td_error(self, model, model_target, action, state, state_next, reward, terminal):
        state_next = tf.expand_dims(state_next, 0)
        state = tf.expand_dims(state, 0)

        q = model.predict(state_next)[0]
        target_q = model_target.predict(state_next)

        if self.DOUBLE:
            max_q = np.argmax(q)
            max_actions = tf.one_hot(max_q, self.ACTION_SPACE)
            q_samp = reward + self.GAMMA * tf.reduce_sum(tf.multiply(target_q, max_actions), axis=1)
        else:
            q_samp = reward + self.GAMMA * tf.reduce_max(target_q, axis=1)      # Bellman Equation

        # arg_max_q = np.argmax(q)
        # q_samp = reward + self.GAMMA * tf.reduce_max(target_q, axis=1)
        q_samp = q_samp * (1 - terminal) - terminal
        masks = tf.one_hot(action, self.ACTION_SPACE)

        q_values = model(state)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

        error = abs(q_action - q_samp) + 0.1
        return error

    def evaluate(self, model, log_dir, episode_id):
        info, prev_info, stack, state = self.SANDBOX.reset_stack(self.ENV)

        episode_reward = 0
        frame_count = 0
        frames = []

        while not self.ENV.is_episode_finished():

            # Capture gameplay experience
            frames = capture(self.ENV, self.SANDBOX, frames)

            # Predict action Q-values from environment state and take best action
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()

            # Apply the sampled action in our environment
            state_next, reward, terminal, info = self.SANDBOX.step(self.ENV, stack, prev_info, action, self.ACTION_SPACE)

            episode_reward += reward

            prev_info = info
            state = state_next
            frame_count += 1

            if terminal:
                break

        render_gif(frames, log_dir + "/loop_" + str(episode_id) + "_" + str(episode_reward))
        return episode_reward
