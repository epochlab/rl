#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from wrappers.doom import Sandbox
from agent import DQNAgent
from memory import ExperienceReplayMemory, PrioritizedReplayMemory
from networks import dqn, dueling_dqn
from utils import load_config, log_feedback

# -----------------------------

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("GPU is", "available" if physical_devices else "NOT AVAILABLE")
print("Eager mode:", tf.executing_eagerly())

# -----------------------------

config = load_config('config.yml')['doom-dqn']
log_dir = "metrics/"

# -----------------------------

sandbox = Sandbox(config)
env, action_space = sandbox.build_env(config['env_name'])

model = dueling_dqn(config['input_shape'], config['window_length'], action_space)
model_target = dueling_dqn(config['input_shape'], config['window_length'], action_space)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

agent = DQNAgent(config, sandbox, env, action_space)

if config['use_per']:
    memory = PrioritizedReplayMemory(config)
else:
    memory = ExperienceReplayMemory(config)

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
info, stack, state = sandbox.reset(env)
prev_info = info

while True:
    action = agent.exploration(frame_count, state, model)
    state_next, reward, terminal, info = sandbox.step(env, stack, prev_info, action, action_space)
    # state_next, reward, terminal, info = sandbox.step(env, action, prev_info)

    if config['use_per']:
        event = (action, state, state_next, reward, terminal)
        td_error = agent.td_error(model, model_target, action, state, state_next, reward, terminal)
        memory.push(event, td_error)
    else:
        memory.push(action, state, state_next, reward, terminal)

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

    if frame_count % config['update_after_actions'] == 0 and frame_count > config['batch_size']:
        loss = agent.learn(memory, model, model_target, optimizer)

    if config['fixed']:
        agent.fixed_target(frame_count, model, model_target)
    else:
        agent.soft_target(model_target.trainable_variables, model.trainable_variables)

    if not config['use_per']:
        memory.limit()

    if terminal:
        print("Frame: {}, Episode: {}, Reward: {}, Loss: {}, Max Life: {}".format(frame_count, episode_count, running_reward, loss, max_life))

    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    if terminal and running_reward > (min_reward + 0.1):
        agent.save(model, model_target, log_dir + timestamp)
        eval_reward = agent.evaluate(model, (log_dir + timestamp), episode_count)
        min_reward = running_reward

    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=episode_count)
        tf.summary.scalar('running_reward', running_reward, step=episode_count)
        tf.summary.scalar('eval_reward', eval_reward, step=episode_count)
        tf.summary.scalar('max_life', max_life, step=episode_count)

    if running_reward == config['min_max'][1]:
        agent.save(model, model_target, log_dir + timestamp)
        print("Solved at episode {}!".format(episode_count))
        break

env.close()
