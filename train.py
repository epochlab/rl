#!/usr/bin/env python3

from baselines.common.atari_wrappers import make_atari, wrap_deepmind

import gym
import os, datetime, imageio

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

from networks.ddqn import q_model

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

MAX_MEMORY_LENGTH = 1000000                     # Maximum replay length
UPDATE_AFTER_ACTIONS = 4                        # Train the model after 4 actions

EPSILON_RANDOM_FRAMES = 50000                   # Number of frames to take random action and observe output
EPSILON_GREEDY_FRAMES = 1000000.0               # Number of frames for exploration

GAMMA = 0.99                                    # Discount factor for past rewards
EPSILON = 1.0                                   # Epsilon greedy parameter
EPSILON_MIN = 0.1                               # Minimum epsilon greedy parameter
EPSILON_MAX = 1.0                               # Maximum epsilon greedy parameter
EPSILON_ANNEALER = (EPSILON_MAX - EPSILON_MIN)  # Rate at which to reduce chance of random action being taken

UPDATE_TARGET_NETWORK = 10000                   # How often to update the target network
TAU = 0.08                                      # Dynamic update factor

DOUBLE = True                                   # Double DQN
DEULING = True                                  # Deuling DQN
PLAYBACK = False                                # Vizualize Training

# -----------------------------

model = q_model(DEULING, INPUT_SHAPE, WINDOW_LENGTH, action_space)
model_target = q_model(DEULING, INPUT_SHAPE, WINDOW_LENGTH, action_space)
# model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

def capture(step, sequence):
    if step < 600:
        frame = env.render(mode='rgb_array')
        sequence.append(frame)

    return sequence

def exploration(eps, nstate, step):
    if frame_count < EPSILON_RANDOM_FRAMES or eps > np.random.rand(1)[0]:
        action = np.random.choice(action_space)
    else:
        state_tensor = tf.convert_to_tensor(nstate)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()

    # NOOP - Fire on first frame of episode
    if step == 0:
        action = 1

    eps -= EPSILON_ANNEALER / EPSILON_GREEDY_FRAMES
    eps = max(eps, EPSILON_MIN)

    return action, eps

def step(naction):
    state_next, reward, terminal, info = env.step(naction)
    state_next = np.array(state_next)

    return state_next, reward, terminal, info

def punish(health, feedback):
    if info['ale.lives'] < health:
        life_lost = True
    else:
        life_lost = feedback
    health = info['ale.lives']

    return life_lost, health

def add_memory(naction, nstate, nstate_next, nterminal, nreward):
    action_history.append(naction)
    state_history.append(nstate)
    state_next_history.append(nstate_next)
    terminal_history.append(nterminal)
    rewards_history.append(nreward)

def sample(memory):
    indices = np.random.choice(range(len(memory)), size=BATCH_SIZE)

    state_sample = np.array([state_history[i] for i in indices])
    state_next_sample = np.array([state_next_history[i] for i in indices])
    rewards_sample = [rewards_history[i] for i in indices]
    action_sample = [action_history[i] for i in indices]
    terminal_sample = tf.convert_to_tensor([float(memory[i]) for i in indices])

    return state_sample, state_next_sample, rewards_sample, action_sample, terminal_sample

@tf.function
def update_target(target_weights, weights, ratio, dynamic):
    if dynamic:
        for (a, b) in zip(target_weights, weights):
            a.assign(b * ratio + a * (1 - ratio))
    else:
        if frame_count % UPDATE_TARGET_NETWORK == 0:
            model_target.set_weights(model.get_weights())

def evaluate(episode_id, instance):
    state = np.array(env.reset())

    episode_reward = 0
    frames = []

    for timestep in range(1, MAX_STEPS_PER_EPISODE):

        # Capture gameplay experience
        frames = capture(timestep, frames)

        # Predict action Q-values from environment state and take best action
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()

        # Apply the sampled action in our environment
        state_next, reward, terminal, _ = env.step(action)
        state = np.array(state_next)

        episode_reward += reward

        if terminal:
            break

    render_gif(frames, log_dir + timestamp + "/" + instance + "_" + str(episode_id) + "_" + str(episode_reward))

    return episode_reward

def render_gif(frames, filename):
    return imageio.mimsave(filename + '.gif', frames)

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print("ID:", timestamp)

log_dir = "logs/fit/"
summary_writer = tf.summary.create_file_writer(log_dir + timestamp)

command = "tensorboard --logdir=" + str(log_dir) + " --port=6006 &"
os.system(command)

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint = tf.train.Checkpoint(model)
checkpoint_path = "training_checkpoints/" + timestamp + "/training_checkpoints"

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
terminal_history = []
episode_reward_history = []

running_reward = 0
episode_count = 0
frame_count = 0

min_reward = -21
eval_reward = -21

while True:  # Run until solved
    state = np.array(env.reset())
    episode_reward = 0

    terminal_life_lost = True
    life = 0

    for timestep in range(1, MAX_STEPS_PER_EPISODE):

        if PLAYBACK:
            env.render();                                                    # View training in real-time

        frame_count += 1

        action, EPSILON = exploration(EPSILON, state, timestep)              # Use epsilon-greedy for exploration

        state_next, reward, terminal, info = step(action)                    # Apply the sampled action in our environment
        terminal_life_lost, life = punish(life, terminal)                    # Punishment for points lost within before terminal state

        add_memory(action, state, state_next, terminal_life_lost, reward)    # Save actions and states in replay buffer

        episode_reward += reward                                             # Update running reward
        state = state_next                                                   # Update state

        #############
        ### TRAIN ###
        #############
        # Update every fourth frame and once batch size is over 32
        if frame_count % UPDATE_AFTER_ACTIONS == 0 and len(terminal_history) > BATCH_SIZE:

            # Sample from replay buffer
            state_sample, state_next_sample, rewards_sample, action_sample, terminal_sample = sample(terminal_history)

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
                loss = keras.losses.Huber()(q_samp, q_action)                              # Calculate loss between new Q-value and old Q-value

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Update the the target network with new weights
        update_target(model_target.trainable_variables, model.trainable_variables, TAU, True)

        # Limit the state and reward history
        if len(rewards_history) > MAX_MEMORY_LENGTH:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del terminal_history[:1]

        # Log details
        if frame_count % MAX_STEPS_PER_EPISODE == 0:
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

        if terminal:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    ############
    ### TEST ###
    ############
    # If running_reward has improved by factor of N; evalute & render without epsilon annealer.
    if running_reward > min_reward:
        checkpoint.save(checkpoint_path)
        eval_reward = evaluate(episode_count, "pong_DQN_test")
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
