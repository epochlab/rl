#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from wrappers.doom import Sandbox
from agent import A2CAgent
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

agent = A2CAgent(config, sandbox, env, action_space)

memory = 123

# -----------------------------

timestamp, summary_writer = log_feedback(log_dir)
print("Job ID:", timestamp)

frame_count = 0
episode_count = 0

loss = 0

episode_reward_history = []
episode_reward = 0
eval_reward = 0
min_reward = 0

life = 0
max_life = 0

# -----------------------------

print("Training...")
info, prev_info, stack, state = sandbox.reset(env)

while not env.is_episode_finished():  # Run until solved
    action, policy = agent.get_action(state, actor)
    state_next, reward, terminal, info = sandbox.step(env, stack, prev_info, action, action_space)          # Apply the sampled action in our environment

    agent.push(action, state, reward)

    if terminal:
        episode_reward = 0
        episode_count += 1

        max_life = max(life, max_life)
        life = 0
    else:
        episode_reward += reward
        life += 1

    prev_info = info
    state = state_next
    frame_count += 1

    if frame_count > config['update_after_actions']:
        agent.learn(actor, critic, action_space)
