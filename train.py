#!/usr/bin/env python3

from baselines.common.atari_wrappers import make_atari, wrap_deepmind

import gym, os, datetime
import numpy as np

import tensorflow as tf
from tensorflow import keras

from agent import agent
from memory import memory
from networks import dqn, dueling_dqn
from utils import log_feedback

print("Eager mode:", tf.executing_eagerly())

# -----------------------------

#print(gym.envs.registry.all())

ENV_NAME = "PongNoFrameskip-v4"
env = make_atari(ENV_NAME)

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(1)

screen_space = env.observation_space
num_states = env.observation_space.shape
action_space = env.action_space.n

print('Frame:', screen_space)
print('States:', num_states[0])
print('Actions:', action_space)

env.unwrapped.get_action_meanings()

# -----------------------------

BATCH_SIZE = 32                                 # Size of batch taken from replay buffer
MAX_STEPS_PER_EPISODE = 18000                   # 5mins at 60fps = 18000 steps

MAX_MEMORY_LENGTH = 1000000                     # Maximum replay length - Train for: 1000000
UPDATE_AFTER_ACTIONS = 4                        # Train the model after 4 actions

GAMMA = 0.99                                    # Discount factor for past rewards
EPSILON = 1.0                                   # Epsilon greedy parameter
TAU = 0.08                                      # Dynamic update factor

DOUBLE = True                                   # Double DQN
DEULING = True                                  # Deuling DQN
PLAYBACK = False                                # Vizualize Training

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

running_reward = 0
eval_reward = 0
min_reward = -21

# -----------------------------

model = dueling_dqn(INPUT_SHAPE, WINDOW_LENGTH, action_space)
model_target = dueling_dqn(INPUT_SHAPE, WINDOW_LENGTH, action_space)
model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

agent = agent(action_space, MAX_STEPS_PER_EPISODE)
memory = memory(BATCH_SIZE, MAX_MEMORY_LENGTH, action_history, state_history, state_next_history, rewards_history, terminal_history)

timestamp, log_dir, summary_writer, checkpoint = log_feedback(model)
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

        state_next, reward, terminal, info = agent.step(env, action)                                    # Apply the sampled action in our environment
        terminal_life_lost, life = agent.punish(info, life, terminal)                                   # Punishment for points lost within before terminal state

        memory.add_memory(action, state, state_next, terminal_life_lost, reward)                        # Save actions and states in replay buffer

        episode_reward += reward                                                                        # Update running reward
        state = state_next                                                                              # Update state
        frame_count += 1

        # Update every fourth frame and once batch size is over 32
        if frame_count % UPDATE_AFTER_ACTIONS == 0 and len(terminal_history) > BATCH_SIZE:

            # Sample from replay buffer
            state_sample, state_next_sample, rewards_sample, action_sample, terminal_sample = memory.sample(terminal_history)

            # Double Q-Learning, decoupling selection and evaluation of the action seletion with the current DQN model.
            q = model.predict(state_next_sample)
            target_q = model_target.predict(state_next_sample)

            # Build the updated Q-values for the sampled future states - DQN / DDQN
            if DOUBLE:
                max_q = tf.argmax(q, axis=1)
                max_actions = tf.one_hot(max_q, action_space)
                q_samp = rewards_sample + GAMMA * tf.reduce_sum(tf.multiply(target_q, max_actions), axis=1)
            else:
                q_samp = rewards_sample + GAMMA * tf.reduce_max(target_q, axis=1)        # Bellman Equation

            q_samp = q_samp * (1 - terminal_sample) - terminal_sample                    # If final frame set the last value to -1
            masks = tf.one_hot(action_sample, action_space)                              # Create a mask so we only calculate loss on the updated Q-values

            with tf.GradientTape() as tape:
                q_values = model(state_sample)                                           # Train the model on the states and updated Q-values
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)           # Apply the masks to the Q-values to get the Q-value for action taken
                loss = keras.losses.Huber()(q_samp, q_action)                            # Calculate loss between new Q-value and old Q-value

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Update the the target network with new weights
        agent.update_target(frame_count, model, model.trainable_variables, model_target, model_target.trainable_variables, TAU, True)

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
        eval_reward = agent.evaluate(model, env, (log_dir + timestamp), episode_count, "pong_DQN_test")
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
