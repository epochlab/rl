import os, random, pylab, cv2
import gym

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from wrappers.gym_atari import Sandbox
from agent import PolicyAgent
from networks import policy_gradient
from utils import load_config

# -----------------------------

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("GPU is", "available" if physical_devices else "NOT AVAILABLE")
print("Eager mode:", tf.executing_eagerly())

# -----------------------------

config = load_config('config.yml')['atari-pg']
log_dir = "metrics/"

# -----------------------------

sandbox = Sandbox(config)
env, action_space = sandbox.build_env(config['env_name'])

agent2 = PolicyAgent(config, sandbox, env, action_space)

class PGAgent:
    # Policy Gradient Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
        # Environment and PG parameters
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.EPISODES, self.max_average = 10000, -21.0 # specific for pong
        self.lr = config['learning_rate']

        self.ROWS = config['input_shape'][0]
        self.COLS = config['input_shape'][1]
        self.REM_STEP = config['window_length']

        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.image_memory = np.zeros(self.state_size)

        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_PG_{}'.format(self.env_name, self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor network model
        self.Actor = policy_gradient(input_shape=self.state_size, action_space = self.action_size, lr=self.lr)

    def load(self, Actor_name):
        self.Actor = load_model(Actor_name, compile=False)

    def save(self):
        self.Actor.save(self.Model_name + '.h5')

    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.plot(self.episodes, self.average, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.savefig(self.path+".png")
            except OSError:
                pass

        return self.average[-1]

    def run(self):
        for e in range(self.EPISODES):
            # state = self.reset()
            state = sandbox.reset(env)
            done, score, SAVING = False, 0, ''
            while not done:
                env.render()
                # Actor picks an action
                model = self.Actor
                action = agent2.act(state, model)
                # Retrieve new state, reward, and whether the state is terminal
                # next_state, reward, done, _ = self.step(action)
                next_state, reward, done, _ = sandbox.step(env, action)
                # Memorize (state, action, reward) for training
                agent2.push(state, action, reward)
                # Update current state
                state = next_state
                score += reward
                if done:
                    average = self.PlotModel(score, e)
                    # saving best models
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                        SAVING = "SAVING"
                    else:
                        SAVING = ""
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))

                    agent2.learn(model)

        # close environemnt when finish training
        self.env.close()

if __name__ == "__main__":
    #env_name = 'Pong-v0'
    env_name = 'PongDeterministic-v4'
    agent = PGAgent(env_name)
    agent.run()
