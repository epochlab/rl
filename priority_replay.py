#!/usr/bin/env python3

import numpy as np
from sum_tree import SumTree

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
            (idx, priority data) = self.TREE.get(s)

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
