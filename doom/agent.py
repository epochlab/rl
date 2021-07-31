#!/usr/bin/env python3

import random
import numpy as np
import tensorflow as tf

from utils import capture, render_gif

class agent:
    def __init__(self, env, action_space, max_steps_per_episode):
        self.ENV = env
        self.ACTION_SPACE = action_space
        self.MAX_STEPS_PER_EPISODE = max_steps_per_episode

        self.EPSILON = 1.0
        self.EPSILON_RANDOM_FRAMES = 5000
        self.EPSILON_GREEDY_FRAMES = 50000.0
        self.EPSILON_MIN = 0.0001
        self.EPSILON_MAX = 1.0
        self.EPSILON_ANNEALER = (self.EPSILON_MAX - self.EPSILON_MIN)

    def exploration(self, frame_count, nstate, model):
        if frame_count < self.EPSILON_RANDOM_FRAMES or self.EPSILON > np.random.rand(1)[0]:
            action = np.zeros([self.ACTION_SPACE])
            select = random.randrange(self.ACTION_SPACE)
            action[select] = 1
            action = action.astype(int)
        else:
            state_tensor = tf.convert_to_tensor(nstate)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()

        self.EPSILON -= self.EPSILON_ANNEALER / self.EPSILON_GREEDY_FRAMES
        self.EPSILON = max(self.EPSILON, self.EPSILON_MIN)
        return action

    def step(self, naction):
        state_next, reward, terminal, info = self.ENV.step(naction)
        return np.array(state_next), reward, terminal, info

    def punish(self, info, health, feedback):
        if info['ale.lives'] < health:
            life_lost = True
        else:
            life_lost = feedback
        health = info['ale.lives']
        return life_lost, health

    def evaluate(self, model, log_dir, episode_id):
        state = np.array(self.ENV.reset())
        episode_reward = 0
        frames = []

        for timestep in range(1, self.MAX_STEPS_PER_EPISODE):

            # Capture gameplay experience
            frames = capture(self.ENV, timestep, frames)

            # Predict action Q-values from environment state and take best action
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()

            # Apply the sampled action in our environment
            state_next, reward, terminal, _ = self.step(action)
            state = np.array(state_next)

            episode_reward += reward

            if terminal:
                break

        render_gif(frames, log_dir + "/loop_" + str(episode_id) + "_" + str(episode_reward))
        return episode_reward
