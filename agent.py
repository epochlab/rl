#!/usr/bin/env python3

import numpy as np

def step(env, naction):
    state_next, reward, terminal, info = env.step(naction)
    state_next = np.array(state_next)
    return state_next, reward, terminal, info
