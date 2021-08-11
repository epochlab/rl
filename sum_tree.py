#!/usr/bin/env python3

import numpy as np

class SumTree:

    def __init__(self, capacity):
        self.FULL = False
        self.WRITE_INDEX = 0
        self.CAPACITY = capacity
        self.TREE = np.zeros(2 * capacity - 1)
        self.DATA = np.zeros(capacity, dtype=object)

    def propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.TREE[parent] += change

        if parent != 0:
            self.propagate(parent, change)

    def retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.TREE):
            return idx

        if s <= self.TREE[left]:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.TREE[left])

    def current_length(self):
        if self.FULL:
            return self.CAPACITY
        else:
            return self.WRITE_INDEX

    def insert(self, data, priority):
        self.DATA[self.WRITE_INDEX] = data

        idx = self.WRITE_INDEX + self.CAPACITY - 1
        self.update(idx, priority)

        self.WRITE_INDEX += 1
        if self.WRITE_INDEX >= self.CAPACITY:
            self.WRITE_INDEX = 0
            self.FULL = True

    def update(self, idx, priority):
        change = priority - self.TREE[idx]

        self.TREE[idx] = priority
        self.propagate(idx, change)

    def get(self, s):
        idx = self.retrieve(0, s)
        data_idx = idx - self.CAPACITY + 1
        return (idx, self.TREE[idx], self.DATA[data_idx])
