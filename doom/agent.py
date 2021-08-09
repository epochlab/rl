#!/usr/bin/env python3

import random
import numpy as np
import tensorflow as tf
from collections import deque

from doom_wrapper import sandbox
from utils import capture, render_gif

class agent:
    def __init__(self, env, action_space):
        self.ENV = env
        self.ACTION_SPACE = action_space

        self.EPSILON = 1.0
        self.EPSILON_RANDOM_FRAMES = 5000
        self.EPSILON_GREEDY_FRAMES = 50000
        self.EPSILON_MIN = 0.0001
        self.EPSILON_MAX = 1.0
        self.EPSILON_ANNEALER = (self.EPSILON_MAX - self.EPSILON_MIN)

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
        self.ENV.new_episode()
        state = self.ENV.get_state()
        info = state.game_variables
        prev_info = info

        frame = state.screen_buffer
        stack = deque([np.zeros((64, 64), dtype=int) for i in range(4)], maxlen=4)
        stack, stack_state = sandbox().framestack(stack, frame, True)

        episode_reward = 0
        frame_count = 0
        frames = []

        while not self.ENV.is_episode_finished():

            # Capture gameplay experience
            frames = capture(self.ENV, frame_count, frames)

            # Predict action Q-values from environment state and take best action
            state_tensor = tf.convert_to_tensor(stack_state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action_idx = tf.argmax(action_probs[0]).numpy()

            action = np.zeros([self.ACTION_SPACE])
            action[action_idx] = 1
            action = action.astype(int)

            # Apply the sampled action in our environment
            next_stack_state, reward, terminated, info = sandbox().step(self.ENV, stack, prev_info, action)

            episode_reward += reward

            prev_info = info
            stack_state = next_stack_state
            frame_count += 1

            if terminated:
                break

        render_gif(frames, log_dir + "/loop_" + str(episode_id) + "_" + str(episode_reward))
        return episode_reward
