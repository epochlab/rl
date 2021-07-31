#!/usr/bin/env python3

import os, datetime, imageio

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

def capture(env, timestep, sequence):
    if timestep < 600:
        frame = env.render(mode='rgb_array')
        sequence.append(frame)
    return sequence

def render_gif(frames, filename):
    return imageio.mimsave(filename + '.gif', frames)

def log_feedback(model, log_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    summary_writer = tf.summary.create_file_writer(log_dir + timestamp)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    os.system("tensorboard --logdir=" + str(log_dir) + " --port=6006 &")
    return timestamp, summary_writer
