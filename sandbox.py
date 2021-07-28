#!/usr/bin/env python3

import random, time
import numpy as np

import gym, vizdoom

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
        img = state.screen_buffer
        misc = state.game_variables
        reward = env.make_action(random.choice(action_space))
        time.sleep(0.02)
    env.close()

atari_search()
