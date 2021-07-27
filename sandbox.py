#!/usr/bin/env python3

import gym
import numpy as np

def atari_search():
    env = gym.make("PongNoFrameskip-v4")
    state = env.reset()
    for i in range(10000):
        env.render()
        action = np.random.choice(env.action_space.n)
        state_next, reward, terminal, info = env.step(action)
