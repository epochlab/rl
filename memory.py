#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

class memory:
    def __init__(self, batch_size, action_history, state_history, state_next_history, rewards_history, terminal_history):
        self.BATCH_SIZE = batch_size
        self.action_history = action_history
        self.state_history = state_history
        self.state_next_history = state_next_history
        self.rewards_history = rewards_history
        self.terminal_history = terminal_history

    def add_memory(self, naction, nstate, nstate_next, nterminal, nreward):
        self.action_history.append(naction)
        self.state_history.append(nstate)
        self.state_next_history.append(nstate_next)
        self.rewards_history.append(nreward)
        self.terminal_history.append(nterminal)

    def sample(self, memory):
        indices = np.random.choice(range(len(memory)), size=self.BATCH_SIZE)

        action_sample = [self.action_history[i] for i in indices]
        state_sample = np.array([self.state_history[i] for i in indices])
        state_next_sample = np.array([self.state_next_history[i] for i in indices])
        rewards_sample = [self.rewards_history[i] for i in indices]
        terminal_sample = tf.convert_to_tensor([float(memory[i]) for i in indices])

        return state_sample, state_next_sample, rewards_sample, action_sample, terminal_sample
