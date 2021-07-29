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

    action_space = env.get_available_buttons_size()
    action_history = []
    fps = 1

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

        action = np.zeros([action_space])
        select = random.randrange(action_space)
        action[select] = 1
        action = action.astype(int)
        env.set_action(action.tolist())
        reward = env.advance_action(fps)
        time.sleep(0.02)
    env.close()
