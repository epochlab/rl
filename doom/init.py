#!/usr/bin/env python3

import numpy as np
import random

import tensorflow as tf

from doom_wrapper import sandbox
from agent import agent
from memory import memory
from networks import dqn, dueling_dqn
from utils import log_feedback

from collections import deque

print("Eager mode:", tf.executing_eagerly())

# -----------------------------

MAX_STEPS_PER_EPISODE = 300                     # 5mins at 60fps = 18000 steps

DOUBLE = True                                   # Double DQN
DYNAMIC = True                                  # Dynamic update
PLAYBACK = False                                # Vizualize Training
FPS = 1

log_dir = "metrics/"

# -----------------------------

# Build sandbox environment
sandbox = sandbox()
env, action_space, INPUT_SHAPE, WINDOW_LENGTH = sandbox.build_env()

# Compile neural networks
model = dueling_dqn(INPUT_SHAPE, WINDOW_LENGTH, action_space)
model_target = dueling_dqn(INPUT_SHAPE, WINDOW_LENGTH, action_space)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Build Agent
agent = agent(env, action_space, MAX_STEPS_PER_EPISODE)
memory = memory(action_space)

# -----------------------------

timestamp, summary_writer = log_feedback(model, log_dir)
print("Job ID:", timestamp)

episode_count = 0

episode_reward_history = []
running_reward = 0
eval_reward = 0
min_reward = 0

GAME = 0
frame_count = 0
max_life = 0
life = 0

life_buffer, ammo_buffer, kills_buffer = [], [], []

# -----------------------------
env.new_episode()
state = env.get_state()
frame = state.screen_buffer
info = state.game_variables
prev_info = info

stack = deque([np.zeros(INPUT_SHAPE, dtype=int) for i in range(4)], maxlen=4)
stack, stack_state = sandbox.framestack(stack, frame, True)

while not env.is_episode_finished():  # Run until solved

    reward = 0

    action = np.zeros([action_space])
    action_idx = agent.exploration(frame_count, state, model)                                         # Use epsilon-greedy for exploration
    action[action_idx] = 1
    action = action.astype(int)

    env.set_action(action.tolist())
    env.advance_action(FPS)

    state = env.get_state()
    terminated = env.is_episode_finished()
    reward = env.get_last_reward()

    if terminated:
        if (life>max_life):
            max_life = life

        GAME +=1

        life_buffer.append(life)
        kills_buffer.append(info[0])
        ammo_buffer.append(info[1])

        print('Game Finished', info)

        env.new_episode()
        state = env.get_state()
        next_frame = state.screen_buffer
        info = state.game_variables

    next_frame = state.screen_buffer
    stack, next_stack_state = sandbox.framestack(stack, next_frame, False)
    info = state.game_variables
    reward = sandbox.shape_reward(reward, info, prev_info)

    if terminated:
        life = 0
    else:
        life += 1

    prev_info = prev_info

    print('Kill:', info[0], 'Ammo:', info[1], 'Reward:', reward)

        # state_next, reward, terminal = sandbox.step(env, action)                                          # Apply the sampled action in our environment
    memory.add_memory(action_idx, stack_state, next_stack_state, reward, terminated)                                  # Save actions and states in replay buffer

    frame_count += 1
    stack_state = next_stack_state

    memory.learn(frame_count, model, model_target, optimizer, DOUBLE)                               # Learn every fourth frame and once batch size is over 32

    if DYNAMIC:                                                                                     # Update the the target network with new weights
        memory.dynamic_target(model_target.trainable_variables, model.trainable_variables)
    else:
        memory.update_target(frame_count, model, model_target)

    memory.limit()                                                                                  # Limit memory cache to defined length

    # if terminal:
    #     break
#
#     # Update running reward to check condition for solving
#     episode_reward_history.append(episode_reward)
#     if len(episode_reward_history) > 100:
#         del episode_reward_history[:1]
#     running_reward = np.mean(episode_reward_history)
#
#     # If running_reward has improved by factor of N; evalute & render without epsilon annealer.
#     if running_reward > min_reward + 1 and episode_count > 10:
#         memory.save(model, model_target, log_dir + timestamp + "/saved_model")
#         eval_reward = agent.evaluate(model, (log_dir + timestamp), episode_count)
#         min_reward = running_reward
#
#     # Feedback
#     with summary_writer.as_default():
#         tf.summary.scalar('running_reward', running_reward, step=episode_count)
#         tf.summary.scalar('eval_reward', eval_reward, step=episode_count)
#
#     # Condition to consider the task solved (Pong = 21)
#     if running_reward == 21:
#         memory.save(model, model_target, log_dir + timestamp + "/saved_model")
#         print("Solved at episode {}!".format(episode_count))
#         break
#
#     episode_count += 1
#
# env.close()
