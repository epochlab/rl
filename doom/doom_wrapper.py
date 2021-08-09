#!/usr/bin/env python3

import vizdoom
import numpy as np
import time, random

import skimage
from skimage import transform, color
from collections import deque

class sandbox:
    def __init__(self):
        self.CONFIG_PATH = '/mnt/vanguard/git/ViZDoom-master/scenarios/defend_the_center.cfg'

        self.INPUT_SHAPE = (64, 64)
        self.WINDOW_LENGTH = 4

    def build_env(self):
        env = vizdoom.DoomGame()
        env.load_config(self.CONFIG_PATH)
        env.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        env.set_window_visible(True)
        env.init()
        action_space = env.get_available_buttons_size()
        return env, action_space, self.INPUT_SHAPE, self.WINDOW_LENGTH

    def preprocess(self, frame, size):
        frame = np.rollaxis(frame, 0, 3)
        frame = skimage.transform.resize(frame, size)
        frame = skimage.color.rgb2gray(frame)
        # frame = frame / 255.0
        return frame

    def framestack(self, stack, state, new_episode):
        frame = self.preprocess(state, self.INPUT_SHAPE)
        if new_episode:
            for _ in range(4):
                stack.append(frame)
        else:
            stack.append(frame)

        stack_state = np.stack(stack, axis=2)
        return stack, stack_state

    def shape_reward(self, reward, info, prev_info):
        if (info[0] > prev_info[0]): # Kill count
            reward = reward + 1

        if (info[1] < prev_info[1]): # Ammo
            reward = reward - 0.1

        if (info[2] < prev_info[2]): # Health
            reward = reward - 0.1

        return reward
