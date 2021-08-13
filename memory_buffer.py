#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from sum_tree import SumTree

class ExperienceReplayMemory:
    def __init__(self, config):
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

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, eps=1e-2):
        self.TREE = SumTree(capacity)
        self.ALPHA = alpha
        self.EPS = eps

    def get_priority(self, td_error):
        return(td_error + self.EPS) ** self.ALPHA

    def current_length(self):
        return self.TREE.current_length()

    def total_sum(self):
        return self.TREE.total_sum()

    def push(self, event, td_error):
        priority = get_priority(td_error)
        self.TREE.insert(event, priority)

    def sample(self, batch_size):
        batch = []
        indices = []
        priorities = []
        segment = self.TREE.total_sum() / batch_size

        for i in range(batch_size):
            l = segment * i
            r = segment * (i+1)

            s = random.uniform(l, r)
            (idx, priority, data) = self.TREE.get(s)

            batch.append(data)
            indicies.append(idx)
            priorities.append(priority)

        samples = map(np.array, zip(*batch))
        return samples, indicies, priorities

    def update(self, idx, td_error):
        if isinstance(idx, list):
            for i in range(len(idx)):
                priority = self.get_priority(td_error[i])
                self.TREE.update(idx[i], priority)
        else:
            priority = self.get_priority(td_error)
            self.TREE.update(idx, priority)
