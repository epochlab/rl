#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from wrappers.gym_atari import Sandbox
from agent import PolicyAgent
from networks import policy_gradient
from utils import load_config

# -----------------------------

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("GPU is", "available" if physical_devices else "NOT AVAILABLE")
print("Eager mode:", tf.executing_eagerly())

# -----------------------------

config = load_config('config.yml')['atari-pg']
log_dir = "metrics/"

# -----------------------------

sandbox = Sandbox(config)
env, action_space = sandbox.build_env(config['env_name'])

model = policy_gradient(input_shape=(config['window_length'], config['input_shape'][0], config['input_shape'][1]), action_space=action_space, lr=config['learning_rate'])
agent = PolicyAgent(config, sandbox, env, action_space)

# -----------------------------

EPISODES = 10000
episode_reward_history = []

frame_count = 0

life = 0
max_life = 0

for e in range(EPISODES):
    state = sandbox.reset(env)
    terminal, episode_reward = False, 0

    while not terminal:
        env.render()
        action = agent.act(state, model)                                    # Actor picks an action
        state_next, reward, terminal, _ = sandbox.step(env, action)         # Retrieve new state, reward, and whether the state is terminal
        agent.push(state, action, reward)                                   # Memorize (state, action, reward) for training

        if terminal:
            agent.learn(model)
            print("Frame {}, Episode: {}/{}, Reward: {}, Average: {:.2f}".format(frame_count, e, EPISODES, episode_reward, running_reward))

            episode_reward = 0
            max_life = max(life, max_life)
            life = 0
        else:
            episode_reward += reward
            life += 1

        state = state_next                                                  # Update current state
        frame_count += 1

        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

env.close()
