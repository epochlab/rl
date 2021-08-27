#!/usr/bin/env python3

import gym
import numpy as np

import skimage
from skimage import transform, color

class Sandbox:
    def __init__(self, config):
        self.INPUT_SHAPE = config['input_shape']
        self.WINDOW_LENGTH = config['window_length']

        self.STACK = np.zeros((config['window_length'], config['input_shape'][0], config['input_shape'][1]))

    def build_env(self, env_name):
        env = gym.make(env_name)
        action_space = env.action_space.n

        print('Frame:', env.observation_space)
        print('States:', env.observation_space.shape[0])
        print('Actions:', action_space)
        env.unwrapped.get_action_meanings()

        return env, action_space

    def preprocess(self, frame):
        frame = frame[35:191]
        frame = skimage.color.rgb2gray(frame)

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
        frame = env.reset()
        for _ in range(self.WINDOW_LENGTH):
            state = self.framestack(frame)
        return state

    def step(self, env, action):
        next_state, reward, terminal, info = env.step(action)
        next_state = self.framestack(next_state)
        return next_state, reward, terminal, info
