#!/usr/bin/env python3

import numpy as np
import skimage
from skimage import transform

from wrappers.doom import Sandbox
from utils import load_config

# -----------------------------

config = load_config()['doom-ddqn']

# -----------------------------

def viewslice(state, count):
    frame = np.array(state)
    frame = processed_frame = np.repeat(frame[:, :, count, np.newaxis], 3, axis=2)
    frame = skimage.transform.resize(frame, dimensions)
    return frame

sandbox = Sandbox(config)
env, action_space = sandbox.build_env(config['env_name'])

info, prev_info, stack, state = sandbox.reset(env)
dimensions = (640, 480)

viewslice(state, 0)
