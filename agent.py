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
        if self.USE_PER:
            samples, indices, priorities = memory.sample(self.BATCH_SIZE)
            action_sample, state_sample, state_next_sample, reward_sample, terminal_sample = samples
        else:
            action_sample, state_sample, state_next_sample, reward_sample, terminal_sample = memory.sample()

        q = model.predict(state_next_sample)
        target_q = model_target.predict(state_next_sample)

        if self.DOUBLE:
            max_q = tf.argmax(q, axis=1)
            max_actions = tf.one_hot(max_q, self.ACTION_SPACE)
            q_samp = reward_sample + self.GAMMA * tf.reduce_sum(tf.multiply(target_q, max_actions), axis=1)
        else:
            q_samp = reward_sample + self.GAMMA * tf.reduce_max(target_q, axis=1)

        q_samp = q_samp * (1 - terminal_sample) - terminal_sample
        masks = tf.one_hot(action_sample, self.ACTION_SPACE)

        with tf.GradientTape() as tape:
            q_values = model(state_sample)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = tf.keras.losses.Huber()(q_samp, q_action)

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
            q_samp = reward + self.GAMMA * tf.reduce_max(target_q, axis=1)

        q_samp = q_samp * (1 - terminal) - terminal
        masks = tf.one_hot(action, self.ACTION_SPACE)

        q_values = model(state)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

        td_error = abs(q_action - q_samp) + 0.1
        return td_error

    def evaluate(self, model, log_dir, episode_id):
        terminal, state, info = self.SANDBOX.reset(self.ENV)
        prev_info = info

        frames = []
        episode_reward = 0

        while True:
            frames = capture(self.ENV, self.SANDBOX, frames)
            action = self.get_action(state, model)
            state_next, reward, terminal, info = self.SANDBOX.step(self.ENV, action, prev_info)

            episode_reward += reward

            prev_info = info
            state = state_next

            if terminal:
                break

        render_gif(frames, log_dir + "/loop_" + str(episode_id) + "_" + str(episode_reward))
        return episode_reward

    def save(self, model, model_target, outdir):
        model.save(outdir + '/model.h5')
        model_target.save(outdir + '/model_target.h5')

class PolicyAgent:
    def __init__(self, config, sandbox, env, action_space):
        self.SANDBOX = sandbox
        self.ENV = env
        self.ACTION_SPACE = action_space

        self.GAMMA = config['gamma']
        self.STATE_SIZE = (config['window_length'], config['input_shape'][0], config['input_shape'][1])

        self.action_history, self.state_history, self.reward_history = [], [], []

    def act(self, state, model):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        policy = model.predict(state_tensor)[0]
        action = np.random.choice(self.ACTION_SPACE, p=policy)
        return action

    def push(self, state, action_idx, reward):
        action = np.zeros([self.ACTION_SPACE])
        action[action_idx] = 1

        self.action_history.append(action)
        self.state_history.append(np.expand_dims(state, axis=0))
        self.reward_history.append(reward)

    def discount_rewards(self):
        sum_reward = 0
        discounted_r = np.zeros_like(self.reward_history)
        for i in reversed(range(0,len(self.reward_history))):
            if self.reward_history[i] != 0:
                sum_reward = 0
            sum_reward = sum_reward * self.GAMMA + self.reward_history[i]
            discounted_r[i] = sum_reward

        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r

    def learn_policy(self, model):
        actions = np.vstack(self.action_history)
        states = np.vstack(self.state_history)

        discounted_r = self.discount_rewards()

        history = model.fit(states, actions, sample_weight=discounted_r, epochs=1, verbose=0)

        self.action_history, self.state_history, self.reward_history = [], [], []
        return history.history['loss'][0]

    def learn_a2c(self, actor, critic):
        states = np.vstack(self.state_history)
        actions = np.vstack(self.action_history)

        values = critic.predict(states)[:, 0]
        discounted_r = self.discount_rewards()
        advantages = discounted_r - values

        actor_history = actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        critic_history = critic.fit(states, discounted_r, epochs=1, verbose=0)

        self.action_history, self.state_history, self.reward_history = [], [], []
        return actor_history.history['loss'][0], critic_history.history['loss'][0]

    def evaluate(self, model, log_dir, episode_id):
        terminal, state, info = self.SANDBOX.reset(self.ENV)
        prev_info = info

        frames = []
        episode_reward = 0

        while not terminal:
            frames = capture(self.ENV, self.SANDBOX, frames)
            action = self.act(state, model)
            state_next, reward, terminal, info = self.SANDBOX.step(self.ENV, action, prev_info)

            prev_info = info
            episode_reward += reward
            state = state_next

            if terminal:
                break

        render_gif(frames, log_dir + "/loop_" + str(episode_id) + "_" + str(episode_reward))
        return episode_reward

    def save(self, model, outdir):
        model.save(outdir + '/model.h5')

class AsynchronousAgent:
    def __init__(self, config, sandbox, env, action_space):
        self.SANDBOX = sandbox
        self.ENV = env
        self.ACTION_SPACE = action_space

        self.GAMMA = config['gamma']
        self.STATE_SIZE = (config['window_length'], config['input_shape'][0], config['input_shape'][1])

    def act(self, state, model):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        policy = model.predict(state_tensor)[0]
        action = np.random.choice(self.ACTION_SPACE, p=policy)
        return action

    def push(self, state, action_idx, reward):
        action = np.zeros([self.ACTION_SPACE])
        action[action_idx] = 1

        self.action_history.append(action)
        self.state_history.append(np.expand_dims(state, axis=0))
        self.reward_history.append(reward)

    def discount_rewards(self, reward):
        sum_reward = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0:
                sum_reward = 0
            sum_reward = sum_reward * self.GAMMA + reward[i]
            discounted_r[i] = sum_reward

        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r

    def learn_a3c(self, actor, critic, actions, states, rewards):
        actions = np.vstack(actions)
        states = np.vstack(states)

        values = critic.predict(states)[:, 0]
        discounted_r = self.discount_rewards(rewards)
        advantages = discounted_r - values

        actor_history = actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        critic_history = critic.fit(states, discounted_r, epochs=1, verbose=0)

        # return actor_history.history['loss'][0], critic_history.history['loss'][0]

    def evaluate(self, model, log_dir, episode_id):
        terminal, state, info = self.SANDBOX.reset(self.ENV)
        prev_info = info

        frames = []
        episode_reward = 0

        while not terminal:
            frames = capture(self.ENV, self.SANDBOX, frames)
            action = self.act(state, model)
            state_next, reward, terminal, info = self.SANDBOX.step(self.ENV, action, prev_info)

            prev_info = info
            episode_reward += reward
            state = state_next

            if terminal:
                break

        render_gif(frames, log_dir + "/loop_" + str(episode_id) + "_" + str(episode_reward))
        return episode_reward

    def save(self, model, outdir):
        model.save(outdir + '/model.h5')
