#!/usr/bin/env python3

import random, time
import numpy as np

import gym, vizdoom

from agent import vision

def atari_search():
    env = gym.make("PongNoFrameskip-v4")
    state = env.reset()
    terminal = False
    while not terminal:
        env.render()
        action = np.random.choice(env.action_space.n)
        state_next, reward, terminal, info = env.step(action)
    env.close()

def doom_search():
    env = vizdoom.DoomGame()
    env.load_config('/mnt/vanguard/git/ViZDoom-master/scenarios/basic.cfg')
    env.init()

    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    action_space = [shoot, left, right]

    env.new_episode()
    while not env.is_episode_finished():
        state = env.get_state()
        n = state.number
        vars = state.game_variables
        screen_buf = state.screen_buffer
        depth_buf = state.depth_buffer
        labels_buf = state.labels_buffer
        automap_buf = state.automap_buffer
        labels = state.labels
        action = random.choice(action_space)
        reward = env.make_action(action)
        time.sleep(0.02)
    env.close()
