#!/usr/bin/env python3

import os, datetime, imageio

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

def capture(env, step, sequence):
    if step < 600:
        frame = env.render(mode='rgb_array')
        sequence.append(frame)
    return sequence

def render_gif(frames, filename):
    return imageio.mimsave(filename + '.gif', frames)

def log(model):
    log_dir = "metrics/"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    summary_writer = tf.summary.create_file_writer(log_dir + timestamp)
    checkpoint = tf.train.Checkpoint(model)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    os.system("tensorboard --logdir=" + str(log_dir) + " --port=6006 &")
    return timestamp, log_dir, summary_writer, checkpoint
