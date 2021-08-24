#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from wrappers.doom import Sandbox
from networks import actor_network, critic_network
from utils import load_config, log_feedback, save, load

# -----------------------------

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("GPU is", "available" if physical_devices else "NOT AVAILABLE")
print("Eager mode:", tf.executing_eagerly())

# -----------------------------

config = load_config('config.yml')['doom-a2c']
log_dir = "metrics/"

# -----------------------------

sandbox = Sandbox(config)
env, action_space = sandbox.build_env(config['env_name'])
value_space = 1

actor = actor_network(config['input_shape'], config['window_length'], action_space)
critic = critic_network(config['input_shape'], config['window_length'], value_space)

optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

actor.summary()
