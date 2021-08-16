#!/usr/bin/env python3

import yaml, os, datetime, imageio
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

def load_config():
    with open('config.yml') as f:
        return yaml.full_load(f)

def capture(env, sandbox, sequence):
    frame = sandbox.render(env)
    sequence.append(frame)
    return(sequence)

def render_gif(frames, filename):
    return imageio.mimsave(filename + '.gif', frames)

def log_feedback(model, log_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    summary_writer = tf.summary.create_file_writer(log_dir + timestamp)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    os.system("tensorboard --logdir=" + str(log_dir) + " --port=6006 &")
    return timestamp, summary_writer

def save(config, model, model_target, outdir):
    model.save(outdir + '/model.h5')
    model_target.save(outdir + '/model_target.h5')

    if not config['use_per']:
        action_history, state_history, state_next_history, reward_history, terminal_history = memory.fetch()
        np.save(outdir + '/action.npy', action_history)
        np.save(outdir + '/state.npy', state_history)
        np.save(outdir + '/state_next.npy', state_next_history)
        np.save(outdir + '/reward.npy', reward_history)
        np.save(outdir + '/terminal.npy', terminal_history)

def load(config, outdir):
    model = tf.keras.models.load_model(outdir + '/model.h5')
    model_target = tf.keras.models.load_model(outdir + '/model_target.h5')

    if not config['use_per']:
        action_history, state_history, state_next_history, reward_history, terminal_history = memory.fetch()
        action_history = np.load(outdir + '/action.npy')
        state_history = np.load(outdir + '/state.npy')
        state_next_history = np.load(outdir + '/state_next.npy')
        reward_history = np.load(outdir + '/reward.npy')
        terminal_history = np.load(outdir + '/terminal.npy')
