#!/usr/bin/env python3

import numpy as np

import tensorflow as tf
from tensorflow import keras

from environment import build_atari
from agent import agent
from memory import memory
from networks import dqn, dueling_dqn
from utils import log_feedback

print("Eager mode:", tf.executing_eagerly())

# -----------------------------

#print(gym.envs.registry.all())

ENV_NAME = "PongNoFrameskip-v4"
env, action_space = build_atari(ENV_NAME)

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

BATCH_SIZE = 32                                 # Size of batch taken from replay buffer
MAX_STEPS_PER_EPISODE = 18000                   # 5mins at 60fps = 18000 steps

MAX_MEMORY_LENGTH = 1000000                     # Maximum replay length - Train for: 1000000

EPSILON = 1.0                                   # Epsilon greedy parameter
UPDATE_AFTER_ACTIONS = 4                        # Train the model after 4 actions

DOUBLE = True                                   # Double DQN
DYNAMIC = True
TAU = 0.08                                      # Dynamic update factor

PLAYBACK = False                                # Vizualize Training

log_dir = "metrics/"

# -----------------------------

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
terminal_history = []
episode_reward_history = []

# -----------------------------

frame_count = 0
episode_count = 0

running_reward = -21
eval_reward = -21
min_reward = -21

# -----------------------------

model = dueling_dqn(INPUT_SHAPE, WINDOW_LENGTH, action_space)
model_target = dueling_dqn(INPUT_SHAPE, WINDOW_LENGTH, action_space)
# model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

agent = agent(env, action_space, MAX_STEPS_PER_EPISODE)
memory = memory(BATCH_SIZE, MAX_MEMORY_LENGTH, action_space, action_history, state_history, state_next_history, rewards_history, terminal_history)

timestamp, summary_writer, checkpoint = log_feedback(model, log_dir)
print("Job ID:", timestamp)

# -----------------------------

while True:  # Run until solved
    state = np.array(env.reset())
    episode_reward = 0

    terminal_life_lost = True
    life = 0

    for timestep in range(1, MAX_STEPS_PER_EPISODE):

        if PLAYBACK:
            env.render();                                                                              # View training in real-time

        action, EPSILON = agent.exploration(EPSILON, model, state, timestep, frame_count)              # Use epsilon-greedy for exploration

        state_next, reward, terminal, info = agent.step(action)                                    # Apply the sampled action in our environment
        terminal_life_lost, life = agent.punish(info, life, terminal)                                   # Punishment for points lost within before terminal state

        memory.add_memory(action, state, state_next, terminal_life_lost, reward)                        # Save actions and states in replay buffer

        episode_reward += reward                                                                        # Update running reward
        state = state_next                                                                              # Update state
        frame_count += 1

        # Update every fourth frame and once batch size is over 32
        if frame_count % UPDATE_AFTER_ACTIONS == 0 and len(terminal_history) > BATCH_SIZE:
            loss = memory.learn(terminal_history, model, model_target, optimizer, DOUBLE)

        # Update the the target network with new weights
        if DYNAMIC:
            agent.dynamic_target(model_target.trainable_variables, model.trainable_variables, TAU)
        else:
            agent.update_target(frame_count, model, model_target)

        memory.limit(len(rewards_history))

        if terminal:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    # If running_reward has improved by factor of N; evalute & render without epsilon annealer.
    if running_reward > min_reward:
        checkpoint.save(log_dir + timestamp + "/saved_models/ckpt")
        eval_reward = agent.evaluate(model, (log_dir + timestamp), episode_count)
        min_reward = running_reward + 1

    # Callbacks
    with summary_writer.as_default():
        tf.summary.scalar('running_reward', running_reward, step=episode_count)
        tf.summary.scalar('eval_reward', eval_reward, step=episode_count)

    # Condition to consider the task solved (Pong = 21)
    if running_reward == 21:
        checkpoint.save(checkpoint_path)
        print("Solved at episode {}!".format(episode_count))
        break

    episode_count += 1

env.close()
