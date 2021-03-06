#!/usr/bin/env python3

import sys
sys.path.append('..')

import numpy as np
import tensorflow as tf

from wrappers.doom import Sandbox
from agent import PolicyAgent
from networks import actor_critic
from utils import load_config, log_feedback

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

actor, critic = actor_critic(config['input_shape'], config['window_length'], action_space, config['learning_rate'])
actor.summary()

agent = PolicyAgent(config, sandbox, env, action_space)

# -----------------------------

timestamp, summary_writer = log_feedback(log_dir)
print("Job ID:", timestamp)

frame_count = 0
episode_count = 0

a_loss, c_loss = 0, 0

episode_reward_history = []
episode_reward = 0
eval_reward = config['min_max'][0]
min_reward = config['min_max'][0]

life = 0
max_life = 0

# -----------------------------

print("Training...")
terminal, state, info = sandbox.reset(env)
prev_info = info

while True:
    action = agent.act(state, actor)
    state_next, reward, terminal, info = sandbox.step(env, action, prev_info)
    agent.push(state, action, reward)

    if terminal:
        a_loss, c_loss = agent.learn_a2c(actor, critic)

        episode_reward = 0
        episode_count += 1

        max_life = max(life, max_life)
        life = 0
    else:
        episode_reward += reward
        life += 1

    prev_info = info
    state = state_next

    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    if terminal:
        print("Frame: {}, Episode: {}, Reward: {}, Actor Loss: {}, Critic Loss: {}, Max Life: {}".format(frame_count, episode_count, running_reward, a_loss, c_loss, max_life))

    with summary_writer.as_default():
        tf.summary.scalar('a_loss', a_loss, step=episode_count)
        tf.summary.scalar('c_loss', c_loss, step=episode_count)
        tf.summary.scalar('running_reward', running_reward, step=episode_count)
        tf.summary.scalar('eval_reward', eval_reward, step=episode_count)
        tf.summary.scalar('max_life', max_life, step=episode_count)

    if terminal and running_reward > (min_reward + 1):
        agent.save(actor, log_dir + timestamp)
        eval_reward = agent.evaluate(actor, (log_dir + timestamp), episode_count)
        min_reward = running_reward

    if running_reward == config['min_max'][1]:
        agent.save(actor, log_dir + timestamp)
        print("Solved at episode {}!".format(episode_count))
        break

    frame_count += 1

env.close()
