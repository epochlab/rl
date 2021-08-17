#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import cv2, skimage
from skimage import transform

from wrappers.doom import Sandbox
from networks import dqn, dueling_dqn
from utils import load_config

# -----------------------------

config = load_config()['doom-ddqn']
log_dir = 'metrics/20210816-233347/'

dim = (640, 480)

# -----------------------------

def load_model(log_dir):
    tf.keras.models.load_model(log_dir + 'model.h5', compile=False)
    return model

def filter_summary(model):
    for layer in model.layers:
        if 'conv' in layer.name:
            filters, biases = layer.get_weights()
            print(layer.name, filters.shape)

def feature_model(model):
    out = model.layers[3].output
    feature_model = tf.keras.Model(inputs = model.inputs, outputs = out)
    return feature_model

def viewslice(state, count):
    frame = np.array(state)
    frame = processed_frame = np.repeat(frame[:, :, count, np.newaxis], 3, axis=2)
    frame = skimage.transform.resize(frame, dim)
    return frame

def heatmap(frame, model):
    with tf.GradientTape() as tape:
        conv_layer = model.get_layer('conv2d_2')
        iterate = tf.keras.models.Model([model.inputs], [model.output, conv_layer.output])
        _model, conv_layer = iterate(frame[np.newaxis, :, :, :])
        _class = _model[:, np.argmax(_model[0])]
        grads = tape.gradient(_class, conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_layer), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap.reshape((7, 7))
        heatmap = cv2.resize(heatmap, dim)
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        return heatmap

# -----------------------------

sandbox = Sandbox(config)

env, action_space = sandbox.build_env(config['env_name'])
info, prev_info, stack, state = sandbox.reset(env)

model = dueling_dqn(config['input_shape'], config['window_length'], action_space)
model = load_model(log_dir)

# -----------------------------

# heatmap = heatmap(state, model)
# cv2.imwrite('heatmap.png', heatmap)
