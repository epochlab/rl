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

def load(config, outdir):
    model = tf.keras.models.load_model(outdir + '/model.h5')
    model_target = tf.keras.models.load_model(outdir + '/model_target.h5')
