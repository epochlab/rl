#!/usr/bin/env python3

import random
import numpy as np
import tensorflow as tf

from doom_wrapper import sandbox
from utils import capture, render_gif

class agent:
    def __init__(self, config, env, action_space):
        self.CONFIG = config
        self.ENV = env
        self.ACTION_SPACE = action_space

        self.EPSILON = config['epsilon']
        self.EPSILON_RANDOM_FRAMES = config['epsilon_random_frames']
        self.EPSILON_GREEDY_FRAMES = config['epsilon_greedy_frames']
        self.EPSILON_MIN = config['epsilon_min']
        self.EPSILON_MAX = config['epsilon_max']
        self.EPSILON_ANNEALER = (config['epsilon_max'] - config['epsilon_min'])

    def exploration(self, frame_count, nstate, model):
        if frame_count < self.EPSILON_RANDOM_FRAMES or self.EPSILON > np.random.rand(1)[0]:
            action_idx = random.randrange(self.ACTION_SPACE)
        else:
            state_tensor = tf.convert_to_tensor(nstate)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action_idx = tf.argmax(action_probs[0]).numpy()

        self.EPSILON -= self.EPSILON_ANNEALER / self.EPSILON_GREEDY_FRAMES
        self.EPSILON = max(self.EPSILON, self.EPSILON_MIN)
        return action_idx

    def evaluate(self, model, log_dir, episode_id):
        info, prev_info, stack, state = sandbox(self.CONFIG).reset(self.ENV)

        episode_reward = 0
        frame_count = 0
        frames = []

        while not self.ENV.is_episode_finished():

            # Capture gameplay experience
            frames = capture(self.ENV, self.CONFIG, frame_count, frames)

            # Predict action Q-values from environment state and take best action
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()

            # Apply the sampled action in our environment
            state_next, reward, terminal, info = sandbox(self.CONFIG).step(self.ENV, stack, prev_info, action, self.ACTION_SPACE)

            episode_reward += reward

            prev_info = info
            state = state_next
            frame_count += 1

            if terminal:
                break

        render_gif(frames, log_dir + "/loop_" + str(episode_id) + "_" + str(episode_reward))
        return episode_reward
