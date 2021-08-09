#!/usr/bin/env python3

import numpy as np

import tensorflow as tf

from doom_wrapper import sandbox
from agent import agent
from memory import memory
from networks import dqn, dueling_dqn
from utils import log_feedback

print("Eager mode:", tf.executing_eagerly())

# -----------------------------

ENV_NAME = '/mnt/vanguard/git/ViZDoom-master/scenarios/defend_the_center.cfg'

DOUBLE = True                                   # Double DQN
DYNAMIC = False                                 # Dynamic update

log_dir = "metrics/"

# -----------------------------

# Build sandbox environment
sandbox = sandbox()
env, action_space, INPUT_SHAPE, WINDOW_LENGTH = sandbox.build_env(ENV_NAME)

# Compile neural networks
model = dueling_dqn(INPUT_SHAPE, WINDOW_LENGTH, action_space)
model_target = dueling_dqn(INPUT_SHAPE, WINDOW_LENGTH, action_space)
# model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Build Agent
agent = agent(env, action_space)
memory = memory(action_space)

# -----------------------------

timestamp, summary_writer = log_feedback(model, log_dir)
print("Job ID:", timestamp)

frame_count = 0
episode_count = 0

episode_reward_history = []
episode_reward = 0
eval_reward = 0
min_reward = 0

# -----------------------------

info, prev_info, stack, stack_state = sandbox.reset(env)

while not env.is_episode_finished():  # Run until solved

    action = np.zeros([action_space])
    action_idx = agent.exploration(frame_count, stack_state, model)                                                   # Use epsilon-greedy for exploration
    action[action_idx] = 1
    action = action.astype(int)

    next_stack_state, reward, terminated, info = sandbox.step(env, stack, prev_info, action)
    memory.add_memory(action_idx, stack_state, next_stack_state, reward, terminated)                                  # Save actions and states in replay buffer

    prev_info = info
    stack_state = next_stack_state
    frame_count += 1

    if terminated:
        episode_reward = 0
        episode_count += 1
    else:
        episode_reward += reward

    memory.learn(frame_count, model, model_target, optimizer, DOUBLE)                                                 # Learn every fourth frame and once batch size is over 32

    if DYNAMIC:                                                                                                       # Update the the target network with new weights
        memory.dynamic_target(model_target.trainable_variables, model.trainable_variables)
    else:
        memory.update_target(frame_count, model, model_target)

    memory.limit()                                                                                                    # Limit memory cache to defined length

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    # If running_reward has improved by factor of N; evalute & render without epsilon annealer.
    if terminated and running_reward > (min_reward + 1):
        memory.save(model, model_target, log_dir + timestamp + "/saved_model")
        eval_reward = agent.evaluate(model, (log_dir + timestamp), episode_count)
        min_reward = running_reward

    # Feedback
    with summary_writer.as_default():
        tf.summary.scalar('running_reward', running_reward, step=episode_count)
        tf.summary.scalar('eval_reward', eval_reward, step=episode_count)

    # Condition to consider the task solved (Pong = 21)
    if running_reward == 100:
        memory.save(model, model_target, log_dir + timestamp + "/saved_model")
        print("Solved at episode {}!".format(episode_count))
        break
