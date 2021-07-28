#!/usr/bin/env python3

import numpy as np

import tensorflow as tf

from doom_wrapper import build_doom
from agent import agent
from memory import memory
from networks import dqn, dueling_dqn
from utils import log_feedback

print("Eager mode:", tf.executing_eagerly())

# -----------------------------

MAX_STEPS_PER_EPISODE = 18000                   # 5mins at 60fps = 18000 steps

DOUBLE = True                                   # Double DQN
DYNAMIC = True                                  # Dynamic update
PLAYBACK = False                                # Vizualize Training

log_dir = "metrics/"

# -----------------------------

env, action_space, INPUT_SHAPE, WINDOW_LENGTH = build_doom(ENV_NAME)

model = dueling_dqn(INPUT_SHAPE, WINDOW_LENGTH, action_space)
model_target = dueling_dqn(INPUT_SHAPE, WINDOW_LENGTH, action_space)
# model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

agent = agent(env, action_space, MAX_STEPS_PER_EPISODE)
memory = memory(action_space)

# -----------------------------

timestamp, summary_writer, checkpoint = log_feedback(model, log_dir)
print("Job ID:", timestamp)

frame_count = 0
episode_count = 0

episode_reward_history = []
running_reward = -21
eval_reward = -21
min_reward = -21

# -----------------------------

while True:  # Run until solved
    state = np.array(env.reset())

    episode_reward = 0
    life = 0
    terminal_life_lost = True

    for timestep in range(1, MAX_STEPS_PER_EPISODE):

        if PLAYBACK:
            env.render();                                                                               # View training in real-time

        action = agent.exploration(frame_count, state, model)                                           # Use epsilon-greedy for exploration
        state_next, reward, terminal, info = agent.step(action)                                         # Apply the sampled action in our environment
        terminal_life_lost, life = agent.punish(info, life, terminal)                                   # Punishment for points lost within before terminal state
        memory.add_memory(action, state, state_next, reward, terminal_life_lost)                        # Save actions and states in replay buffer

        episode_reward += reward                                                                        # Update running reward
        state = state_next                                                                              # Update state
        frame_count += 1

        memory.learn(frame_count, model, model_target, optimizer, DOUBLE)                               # Learn every fourth frame and once batch size is over 32

        if DYNAMIC:                                                                                     # Update the the target network with new weights
            memory.dynamic_target(model_target.trainable_variables, model.trainable_variables)
        else:
            memory.update_target(frame_count, model, model_target)

        memory.limit()                                                                                  # Limit memory cache to defined length

        if terminal:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    # If running_reward has improved by factor of N; evalute & render without epsilon annealer.
    if running_reward > min_reward + 1 and episode_count > 10:
        memory.save(model, model_target, log_dir + timestamp + "/saved_models")
        eval_reward = agent.evaluate(model, (log_dir + timestamp), episode_count)
        min_reward = running_reward

    # Feedback
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
