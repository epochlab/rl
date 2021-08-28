#!/usr/bin/env python3

import gym
import numpy as np

import skimage
from skimage import transform, color

class Sandbox:
    def __init__(self, config):
        self.INPUT_SHAPE = config['input_shape']
        self.WINDOW_LENGTH = config['window_length']
        self.GRADE = config['grade']
        self.VISIBLE = config['visible']

        self.STACK = np.zeros((config['window_length'], config['input_shape'][0], config['input_shape'][1]))

    def build_env(self, env_name):
        env = gym.make(env_name)
        action_space = env.action_space.n
        return env, action_space

    def preprocess(self, frame):
        frame = skimage.color.rgb2gray(frame)

        if self.GRADE:
            frame = frame[35:191]
            frame[frame < 0.5] = 0
            frame[frame >= 0.5] = 255

        frame = skimage.transform.resize(frame, self.INPUT_SHAPE)
        return frame

    def framestack(self, state):
        frame = self.preprocess(state)
        frame = np.array(frame).astype(np.float32) / 255.0

        self.STACK = np.roll(self.STACK, 1, axis=0)
        self.STACK[0,:,:] = frame
        return np.expand_dims(self.STACK, axis=0)

    def reset(self, env):
        terminal = False
        frame = env.reset()
        for _ in range(self.WINDOW_LENGTH):
            state = self.framestack(frame)
        return terminal, state

    def step(self, env, action):
        if self.VISIBLE:
            env.render()

        next_state, reward, terminal, info = env.step(action)
        next_state = self.framestack(next_state)
        reward = self.shape_reward(reward)

        if terminal:
            _, state = self.reset(env)
            terminal = True

        return next_state, reward, terminal, info

    def shape_reward(self, reward):
        reward = np.sign(reward)
        return reward

    def view_human(self, env):
        frame = env.render(mode='rgb_array')
        return frame
