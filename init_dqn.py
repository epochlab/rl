#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from doom_wrapper import sandbox
from agent import agent
from memory import memory
from networks import dqn, dueling_dqn
from utils import log_feedback

print("Eager mode:", tf.executing_eagerly())

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("GPU is", "available" if physical_devices else "NOT AVAILABLE")

# -----------------------------

ENV_NAME = '/mnt/vanguard/git/ViZDoom-master/scenarios/defend_the_center.cfg'

DOUBLE = True                                   # Double DQN
DYNAMIC = False                                 # Dynamic update

log_dir = "metrics/"

# -----------------------------

# Build sandbox environment
sandbox = sandbox()
env, action_space, input_shape, window_length = sandbox.build_env(ENV_NAME)

# Compile neural networks
model = dueling_dqn(input_shape, window_length, action_space)
model_target = dueling_dqn(input_shape, window_length, action_space)
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

info, prev_info, stack, state = sandbox.reset(env)

while not env.is_episode_finished():  # Run until solved

    action = agent.exploration(frame_count, state, model)                                                   # Use epsilon-greedy for exploration
    state_next, reward, terminal, info = sandbox.step(env, stack, prev_info, action, action_space)          # Apply the sampled action in our environment
    memory.add_memory(action, state, state_next, reward, terminal)                                          # Save actions and states in replay buffer

    prev_info = info
    state = state_next
    frame_count += 1

    if terminal:
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
    if terminal and running_reward > (min_reward + 1):
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
