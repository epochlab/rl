#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

import threading, time
from threading import Thread, Lock

from wrappers.doom import Sandbox
from agent import AsynchronousAgent
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

agent = AsynchronousAgent(config, sandbox, env, action_space)

lock = Lock()

# -----------------------------

def execute():
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
    terminal, state, info, image_memory = sandbox.async_reset(env)
    prev_info = info

    actions, states, rewards = [], [], []

    while True:
        action = agent.act(state, actor)
        state_next, reward, terminal, info = sandbox.async_step(env, action, prev_info, image_memory)
        actions.append(sandbox.one_hot(env, action))
        states.append(tf.expand_dims(state_next, 0))
        rewards.append(reward)

        if terminal:
            a_loss, c_loss = agent.learn_a3c(actor, critic, actions, states, rewards)

            actions, states, rewards = [], [], []

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

        frame_count += 1

    env.close()

# -----------------------------

def async_train(n_threads):
    env.close()
    envs = [sandbox.build_env(config['env_name'])[0] for i in range(n_threads)]

    threads = [threading.Thread(target=train_threading, daemon=True, args=(envs[i], i)) for i in range(n_threads)]

    for t in threads:
        time.sleep(2)
        t.start()

    for t in threads:
        time.sleep(10)
        t.join()

def train_threading(env, thread):
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

    lock = Lock()

    # -----------------------------

    print("Training...")
    terminal, state, info, image_memory = sandbox.async_reset(env)
    prev_info = info

    actions, states, rewards = [], [], []

    while True:
        action = agent.act(state, actor)
        state_next, reward, terminal, info = sandbox.async_step(env, action, prev_info, image_memory)
        actions.append(sandbox.one_hot(env, action))
        states.append(tf.expand_dims(state_next, 0))
        rewards.append(reward)

        if terminal:
            lock.acquire()
            a_loss, c_loss = agent.learn_a3c(actor, critic, actions, states, rewards)
            lock.release()

            actions, states, rewards = [], [], []

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
            print("Frame: {}, Episode: {}, Thread: {}, Reward: {}, Actor Loss: {}, Critic Loss: {}, Max Life: {}".format(frame_count, episode_count, thread, running_reward, a_loss, c_loss, max_life))

        with summary_writer.as_default():
            tf.summary.scalar('a_loss', a_loss, step=episode_count)
            tf.summary.scalar('c_loss', c_loss, step=episode_count)
            tf.summary.scalar('running_reward', running_reward, step=episode_count)
            tf.summary.scalar('eval_reward', eval_reward, step=episode_count)
            tf.summary.scalar('max_life', max_life, step=episode_count)

        frame_count += 1

    env.close()

# -----------------------------

# execute()
async_train(n_threads=3)
