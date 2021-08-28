#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from wrappers.gym_atari import Sandbox
from agent import PolicyAgent
from networks import policy_gradient
from utils import load_config, log_feedback, save

# -----------------------------

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("GPU is", "available" if physical_devices else "NOT AVAILABLE")
print("Eager mode:", tf.executing_eagerly())

# -----------------------------

config = load_config('config.yml')['atari-policy']
log_dir = "metrics/"

# -----------------------------

sandbox = Sandbox(config)
env, action_space = sandbox.build_env(config['env_name'])

model = policy_gradient(config['input_shape'], config['window_length'], action_space, config['learning_rate'])
model.summary()

agent = PolicyAgent(config, sandbox, env, action_space)

# -----------------------------

timestamp, summary_writer = log_feedback(log_dir)
print("Job ID:", timestamp)

frame_count = 0
episode_count = 0

loss = 0

episode_reward_history = []
episode_reward = 0
eval_reward = config['min_max'][0]
min_reward = config['min_max'][0]

life = 0
max_life = 0

# -----------------------------

print("Training...")
terminal, state = sandbox.reset(env)

while True:
    action = agent.act(state, model)
    state_next, reward, terminal, info = sandbox.step(env, action)
    agent.push(state, action, reward)

    if terminal:
        loss = agent.learn(model)

        episode_reward = 0
        episode_count += 1

        max_life = max(life, max_life)
        life = 0
    else:
        episode_reward += reward
        life += 1

    state = state_next
    frame_count += 1

    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    if terminal:
        print("Frame: {}, Episode: {}, Reward: {}, Loss: {}, Max Life: {}".format(frame_count, episode_count, running_reward, loss, max_life))

    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=episode_count)
        tf.summary.scalar('running_reward', running_reward, step=episode_count)
        tf.summary.scalar('eval_reward', eval_reward, step=episode_count)
        tf.summary.scalar('max_life', max_life, step=episode_count)

    if terminal and running_reward > (min_reward + 1):
        agent.save(model, log_dir + timestamp)
        eval_reward = agent.evaluate(model, (log_dir + timestamp), episode_count)
        min_reward = running_reward

    if running_reward == config['min_max'][1]:
        agent.save(model, log_dir + timestamp)
        print("Solved at episode {}!".format(episode_count))
        break

env.close()
