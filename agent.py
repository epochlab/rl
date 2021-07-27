#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from utils import capture, render_gif

class agent:
    def __init__(self, action_space, max_steps_per_episode):
        self.ACTION_SPACE = action_space
        self.MAX_STEPS_PER_EPISODE = max_steps_per_episode

        self.EPSILON_RANDOM_FRAMES = 50000
        self.EPSILON_GREEDY_FRAMES = 1000000.0
        self.EPSILON_MIN = 0.1
        self.EPSILON_MAX = 1.0
        self.EPSILON_ANNEALER = (self.EPSILON_MAX - self.EPSILON_MIN)
        self.UPDATE_TARGET_NETWORK = 10000

    def exploration(self, eps, model, nstate, step, frame_count):
        if frame_count < self.EPSILON_RANDOM_FRAMES or eps > np.random.rand(1)[0]:
            action = np.random.choice(self.ACTION_SPACE)
        else:
            state_tensor = tf.convert_to_tensor(nstate)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()

        # NOOP - Fire on first frame of episode
        if step == 0:
            action = 1

        eps -= self.EPSILON_ANNEALER / self.EPSILON_GREEDY_FRAMES
        eps = max(eps, self.EPSILON_MIN)
        return action, eps

    def step(self, env, naction):
        state_next, reward, terminal, info = env.step(naction)
        return np.array(state_next), reward, terminal, info

    def punish(self, info, health, feedback):
        if info['ale.lives'] < health:
            life_lost = True
        else:
            life_lost = feedback
        health = info['ale.lives']
        return life_lost, health

    def evaluate(self, model, env, log_dir, episode_id, instance):
        state = np.array(env.reset())
        episode_reward = 0
        frames = []

        for timestep in range(1, self.MAX_STEPS_PER_EPISODE):

            # Capture gameplay experience
            frames = capture(env, timestep, frames)

            # Predict action Q-values from environment state and take best action
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()

            # Apply the sampled action in our environment
            state_next, reward, terminal, _ = self.step(env, action)
            state = np.array(state_next)

            episode_reward += reward

            if terminal:
                break

        render_gif(frames, log_dir + "/" + instance + "_" + str(episode_id) + "_" + str(episode_reward))
        return episode_reward

    def update_target(self, frame_count, model, weights, model_target, target_weights, tau, dynamic):
        if dynamic:
            for (a, b) in zip(target_weights, weights):
                a.assign(b * tau + a * (1 - tau))
        else:
            if frame_count % self.UPDATE_TARGET_NETWORK == 0:
                model_target.set_weights(model.get_weights())
