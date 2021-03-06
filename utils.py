#!/usr/bin/env python3

import yaml, os, datetime, imageio
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

def load_config(file):
    with open(file) as f:
        return yaml.full_load(f)

def capture(env, sandbox, sequence):
    frame = sandbox.view_human(env)
    sequence.append(frame)
    return sequence

def render_gif(frames, filename):
    frames = np.uint8(frames)
    return imageio.mimsave(filename + '.gif', frames)

def log_feedback(log_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    summary_writer = tf.summary.create_file_writer(log_dir + timestamp)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    os.system("tensorboard --logdir=" + str(log_dir) + " --port=6006 &")
    return timestamp, summary_writer

def load(outdir, id):
    model = tf.keras.models.load_model(outdir + '/model_' + str(id) + '.h5', compile=False)
    return model
