#!/usr/bin/env python3

import vizdoom

CONFIG_PATH = '/mnt/vanguard/git/ViZDoom-master/scenarios/basic.cfg'
MAP = 'map01'

def build_doom():
    INPUT_SHAPE = (64, 64)
    WINDOW_LENGTH = 4

    env = vizdoom.DoomGame()
    env.load_config(CONFIG_PATH)
    env.init()

    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    action_space = [shoot, left, right]

    return env, action_space, INPUT_SHAPE, WINDOW_LENGTH
