#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

class memory:
    def __init__(self, action_space):
        self.ACTION_SPACE = action_space

        self.BATCH_SIZE = 32
        self.MAX_MEMORY_LENGTH = 1000000
        self.UPDATE_AFTER_ACTIONS = 4
        self.GAMMA = 0.99
        self.UPDATE_TARGET_NETWORK = 10000
        self.TAU = 0.08

        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.reward_history = []
        self.terminal_history = []

    def add_memory(self, naction, nstate, nstate_next, nreward, nterminal):
        self.action_history.append(naction)
        self.state_history.append(nstate)
        self.state_next_history.append(nstate_next)
        self.reward_history.append(nreward)
        self.terminal_history.append(nterminal)

    def sample(self, memory):
        indices = np.random.choice(range(len(memory)), size=self.BATCH_SIZE)

        action_sample = [self.action_history[i] for i in indices]
        state_sample = np.array([self.state_history[i] for i in indices])
        state_next_sample = np.array([self.state_next_history[i] for i in indices])
        reward_sample = [self.reward_history[i] for i in indices]
        terminal_sample = tf.convert_to_tensor([float(memory[i]) for i in indices])
        return state_sample, state_next_sample, reward_sample, action_sample, terminal_sample

    def limit(self):
        if len(self.terminal_history) > self.MAX_MEMORY_LENGTH:
            del self.action_history[:1]
            del self.state_history[:1]
            del self.state_next_history[:1]
            del self.reward_history[:1]
            del self.terminal_history[:1]

    def learn(self, frame_count, model, model_target, optimizer, double):
        if frame_count % self.UPDATE_AFTER_ACTIONS == 0 and len(self.terminal_history) > self.BATCH_SIZE:
            # Sample from replay buffer
            state_sample, state_next_sample, reward_sample, action_sample, terminal_sample = self.sample(self.terminal_history)

            # Double Q-Learning, decoupling selection and evaluation of the action seletion with the current DQN model.
            q = model.predict(state_next_sample)
            target_q = model_target.predict(state_next_sample)

            # Build the updated Q-values for the sampled future states - DQN / DDQN
            if double:
                max_q = tf.argmax(q, axis=1)
                max_actions = tf.one_hot(max_q, self.ACTION_SPACE)
                q_samp = reward_sample + self.GAMMA * tf.reduce_sum(tf.multiply(target_q, max_actions), axis=1)
            else:
                q_samp = reward_sample + self.GAMMA * tf.reduce_max(target_q, axis=1)      # Bellman Equation

            q_samp = q_samp * (1 - terminal_sample) - terminal_sample                       # If final frame set the last value to -1
            masks = tf.one_hot(action_sample, self.ACTION_SPACE)                            # Create a mask so we only calculate loss on the updated Q-values

            with tf.GradientTape() as tape:
                q_values = model(state_sample)                                              # Train the model on the states and updated Q-values
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)              # Apply the masks to the Q-values to get the Q-value for action taken
                loss = tf.keras.losses.Huber()(q_samp, q_action)                            # Calculate loss between new Q-value and old Q-value

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

    def update_target(self, frame_count, model, model_target):
        if frame_count % self.UPDATE_TARGET_NETWORK == 0:
            model_target.set_weights(model.get_weights())

    def dynamic_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.TAU + a * (1 - self.TAU))

    def save(self, outdir):
        np.save(outdir + '/action.npy', self.action_history)
        np.save(outdir + '/state.npy', self.state_history)
        np.save(outdir + '/state_next.npy', self.state_next_history)
        np.save(outdir + '/reward.npy', self.reward_history)
        np.save(outdir + '/terminal.npy', self.terminal_history)

    def load(self, outdir):
        self.action_history = np.load(outdir + '/action.npy')
        self.state_history = np.load(outdir + '/state.npy')
        self.state_next_history = np.load(outdir + '/state_next.npy')
        self.reward_history = np.load(outdir + '/reward.npy')
        self.terminal_history = np.load(outdir + '/terminal.npy')
