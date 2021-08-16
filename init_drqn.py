#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from wrappers.doom import Sandbox
from agent import Agent
from memory import ExperienceReplayMemory, PrioritizedReplayMemory
from networks import dqn, dueling_dqn, drqn
from utils import load_config, log_feedback, save, load

# -----------------------------

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("GPU is", "available" if physical_devices else "NOT AVAILABLE")
print("Eager mode:", tf.executing_eagerly())

# -----------------------------

config = load_config()['doom-drqn']
log_dir = "metrics/"

# -----------------------------

sandbox = Sandbox(config)
env, action_space = sandbox.build_env(config['env_name'])

state_size = (config['trace_length'], config['input_shape'][0], config['input_shape'][1], config['img_channels'])
model = drqn(state_size, action_space)
model_target = drqn(state_size, action_space)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

agent = Agent(config, sandbox, env, action_space)

if config['use_per']:
    memory = PrioritizedReplayMemory(config)
else:
    memory = ExperienceReplayMemory(config)

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

# info, prev_info, stack, state = sandbox.reset(env)
#
# while not env.is_episode_finished():  # Run until solved
#
#     action = agent.exploration(frame_count, state, model)                                                   # Use epsilon-greedy for exploration
#     state_next, reward, terminal, info = sandbox.step(env, stack, prev_info, action, action_space)          # Apply the sampled action in our environment
#
#     if config['use_per']:
#         event = (action, state, state_next, reward, terminal)                                               # PrioritizedReplayMemory
#         td_error = agent.td_error(model, model_target, action, state, state_next, reward, terminal)
#         memory.push(event, td_error)
#     else:
#         memory.push(action, state, state_next, reward, terminal)                                            # Save actions and states to ExperienceReplayMemory
#
#     prev_info = info
#     state = state_next
#     frame_count += 1
#
#     if terminal:
#         episode_reward = 0
#         episode_count += 1
#     else:
#         episode_reward += reward
#
#     agent.learn(frame_count, memory, model, model_target, optimizer)                                       # Learn every fourth frame and once batch size is over 32
#
#     if config['fixed_q']:                                                                                  # Update the the target network with new weights
#         agent.fixed_q(model_target.trainable_variables, model.trainable_variables)
#     else:
#         agent.static_target(frame_count, model, model_target)
#
#     if not config['use_per']:
#         memory.limit()                                                                                     # Limit memory cache to defined length
#
#     # Update running reward to check condition for solving
#     episode_reward_history.append(episode_reward)
#     if len(episode_reward_history) > 100:
#         del episode_reward_history[:1]
#     running_reward = np.mean(episode_reward_history)
#
#     # If running_reward has improved by factor of N; evalute & render without epsilon annealer.
#     if terminal and running_reward > (min_reward + 1):
#         save(config, model, model_target, log_dir + timestamp + "/saved_model")
#         eval_reward = agent.evaluate(model, (log_dir + timestamp), episode_count)
#         min_reward = running_reward
#
#     # Feedback
#     with summary_writer.as_default():
#         tf.summary.scalar('running_reward', running_reward, step=episode_count)
#         tf.summary.scalar('eval_reward', eval_reward, step=episode_count)
#
#     # Condition to consider the task solved (Pong = 21)
#     if running_reward == 100:
#         save(config, model, model_target, log_dir + timestamp + "/saved_model")
#         print("Solved at episode {}!".format(episode_count))
#         break
