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
        self.FPS = config['fps']
        self.STACK = np.zeros((config['input_shape'][0], config['input_shape'][1], config['window_length']))

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
        frame = np.array(frame).astype(np.float32) / 255.0
        return frame

    def framestack(self, state):
        frame = self.preprocess(state)
        self.STACK = np.roll(self.STACK, 1, axis=2)
        self.STACK[:,:,0] = frame
        return self.STACK

    def reset(self, env):
        frame = env.reset()
        terminal = False
        info = None
        for _ in range(self.WINDOW_LENGTH):
            state = self.framestack(frame)
        return terminal, state, info

    def step(self, env, action, prev_info):
        if self.VISIBLE:
            env.render()

        total_reward = 0.0
        buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        for i in range(self.FPS):
            next_state, reward, terminal, info = env.step(action)
            if i == self.FPS - 2: buffer[0] = next_state
            if i == self.FPS - 1: buffer[1] = next_state
            total_reward += reward
            if terminal:
                _, state, info = self.reset(env)
                terminal = True
                break

        max_frame = buffer.max(axis=0)
        next_state = self.framestack(max_frame)

        total_reward = self.shape_reward(total_reward)
        return next_state, total_reward, terminal, info

    def shape_reward(self, reward):
        reward = np.sign(reward)
        return reward

    def view_human(self, env):
        frame = env.render(mode='rgb_array')
        return frame
