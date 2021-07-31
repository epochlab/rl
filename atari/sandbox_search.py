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
    env.set_window_visible(True)

    action_space = env.get_available_buttons_size()
    action_history = []
    FPS = 1

    env.new_episode()
    while not env.is_episode_finished():
        state = env.get_state()
        n = state.number
        info = state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
        prev_info = info

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
        env.advance_action(FPS)

        state = env.get_state()
        terminated = env.is_episode_finished()
        reward = env.get_last_reward()

        time.sleep(0.02)
    env.close()
