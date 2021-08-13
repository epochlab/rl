#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

class memory:
    def __init__(self, config, action_space):
        self.ACTION_SPACE = action_space

        self.BATCH_SIZE = config['batch_size']
        self.MAX_MEMORY_LENGTH = config['max_memory_length']

        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.reward_history = []
        self.terminal_history = []

    def add_memory(self, action, state, state_next, reward, terminal):
        self.action_history.append(action)
        self.state_history.append(state)
        self.state_next_history.append(state_next)
        self.reward_history.append(reward)
        self.terminal_history.append(terminal)

    def sample(self):
        indices = np.random.choice(range(len(self.terminal_history)), size=self.BATCH_SIZE)

        action_sample = [self.action_history[i] for i in indices]
        state_sample = np.array([self.state_history[i] for i in indices])
        state_next_sample = np.array([self.state_next_history[i] for i in indices])
        reward_sample = [self.reward_history[i] for i in indices]
        terminal_sample = tf.convert_to_tensor([float(self.terminal_history[i]) for i in indices])
        return action_sample, state_sample, state_next_sample, reward_sample, terminal_sample

    def limit(self):
        if len(self.terminal_history) > self.MAX_MEMORY_LENGTH:
            del self.action_history[:1]
            del self.state_history[:1]
            del self.state_next_history[:1]
            del self.reward_history[:1]
            del self.terminal_history[:1]

    def fetch(self):
        return self.action_history, self.state_history, self.state_next_history, self.reward_history, self.terminal_history
