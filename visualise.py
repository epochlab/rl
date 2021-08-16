#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import cv2, skimage, imageio
from skimage import transform

from wrappers.doom import Sandbox
from networks import dqn, dueling_dqn
from utils import load_config

# -----------------------------

config = load_config()['doom-ddqn']
log_dir = 'metrics/20210816-233347/'

dimensions = (640, 480)

# -----------------------------

def load_model(log_dir):
    tf.keras.models.load_model(log_dir + 'model.h5', compile=False)
    return model

def viewslice(state, count):
    frame = np.array(state)
    frame = processed_frame = np.repeat(frame[:, :, count, np.newaxis], 3, axis=2)
    frame = skimage.transform.resize(frame, dimensions)
    return frame

def heatmap(frame, model):
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv2d_2')
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(frame[np.newaxis, :, :, :])
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap.reshape((7, 7))
        heatmap = skimage.transform.resize(heatmap, config['input_shape'])
        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET) / 255.0
        return heatmap

# -----------------------------

sandbox = Sandbox(config)

env, action_space = sandbox.build_env(config['env_name'])
info, prev_info, stack, state = sandbox.reset(env)

model = dueling_dqn(config['input_shape'], config['window_length'], action_space)
model = load_model(log_dir)

# -----------------------------

heatmap = heatmap(state, model)
