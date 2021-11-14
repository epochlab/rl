#!/usr/bin/env python3

import random
import numpy as np
import tensorflow as tf

from utils import capture, render_gif

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
