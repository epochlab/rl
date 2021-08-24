#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from wrappers.doom import Sandbox
from agent import DQNAgent
from memory import ExperienceReplayMemory, PrioritizedReplayMemory
from networks import dqn, dueling_dqn
from utils import load_config, log_feedback, save, load

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
eval_reward = 0
min_reward = 0

life = 0
max_life = 0

# -----------------------------

print("Training...")
info, prev_info, stack, state = sandbox.reset(env)

while not env.is_episode_finished():  # Run until solved

    action = agent.exploration(frame_count, state, model)                                                   # Use epsilon-greedy for exploration
    state_next, reward, terminal, info = sandbox.step(env, stack, prev_info, action, action_space)          # Apply the sampled action in our environment

    if config['use_per']:
        event = (action, state, state_next, reward, terminal)                                               # PrioritizedReplayMemory
        td_error = agent.td_error(model, model_target, action, state, state_next, reward, terminal)
        memory.push(event, td_error)
    else:
        memory.push(action, state, state_next, reward, terminal)                                            # Save actions and states to ExperienceReplayMemory

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

    if frame_count % config['update_after_actions'] == 0 and frame_count > config['batch_size']:           # Learn every fourth frame and once batch size is over 32
        loss = agent.learn(memory, model, model_target, optimizer)

    if config['fixed']:                                                                                    # Update the the target network with new weights
        agent.fixed_target(frame_count, model, model_target)
    else:
        agent.soft_target(model_target.trainable_variables, model.trainable_variables)

    if not config['use_per']:
        memory.limit()                                                                                     # Limit memory cache to defined length

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    # If running_reward has improved by factor of N; evalute & render without epsilon annealer.
    if terminal and running_reward > (min_reward + 0.1):
        save(model, model_target, log_dir + timestamp)
        eval_reward = agent.evaluate(model, (log_dir + timestamp), episode_count)
        min_reward = running_reward

    # Feedback
    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=episode_count)
        tf.summary.scalar('running_reward', running_reward, step=episode_count)
        tf.summary.scalar('eval_reward', eval_reward, step=episode_count)
        tf.summary.scalar('max_life', max_life, step=episode_count)

    # # Condition to consider the task solved
    if running_reward == 25:                                                    # Pong: 21 | Breakdout: 40 | Doom (Defend the Center): 100 | Doom (Deadly Corridor): 25
        save(model, model_target, log_dir + timestamp)
        print("Solved at episode {}!".format(episode_count))
        break
