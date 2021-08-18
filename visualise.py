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

def view_human(env):
    frame = sandbox.render(env)
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_RGB

def view_machine(state):
    state = np.array(state) * 255.0
    state = cv2.resize(state, (state.shape[0]*2, state.shape[1]*2))

    x0 = np.repeat(state[:, :, 0, np.newaxis], 3, axis=2)
    x1 = np.repeat(state[:, :, 1, np.newaxis], 3, axis=2)
    x2 = np.repeat(state[:, :, 2, np.newaxis], 3, axis=2)
    x3 = np.repeat(state[:, :, 3, np.newaxis], 3, axis=2)

    grid = np.concatenate((x0, x1, x2, x3), axis=1)
    return grid

def attention_window(frame, model, heatmap):
    with tf.GradientTape() as tape:
        conv_layer = model.get_layer('conv2d')
        iterate = tf.keras.models.Model([model.inputs], [model.output, conv_layer.output])
        _model, conv_layer = iterate(frame[np.newaxis, :, :, :])
        _class = _model[:, np.argmax(_model[0])]
        grads = tape.gradient(_class, conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        attention = tf.reduce_mean(tf.multiply(pooled_grads, conv_layer), axis=-1)

        atten_map = np.maximum(attention, 0) / np.max(attention)
        atten_map = atten_map.reshape((20, 20))
        atten_map = cv2.resize(atten_map, dim, interpolation=cv2.INTER_AREA)
        atten_map = np.uint8(atten_map * 255.0)

        if heatmap:
            heatmap = cv2.applyColorMap(atten_map, cv2.COLORMAP_TURBO)
            return heatmap
        else:
            atten_map = np.expand_dims(atten_map, axis=0)
            return atten_map

def attention_comp():
    human = view_human(env)
    attention = attention_window(state, model, False)

    mask = np.zeros_like(human)
    mask[:,:,0] = attention
    mask[:,:,1] = attention
    mask[:,:,2] = attention

    return human * (mask / 255.0)

# -----------------------------

sandbox = Sandbox(config)

env, action_space = sandbox.build_env(config['env_name'])
info, prev_info, stack, state = sandbox.reset(env)

model = dueling_dqn(config['input_shape'], config['window_length'], action_space)
model = load_model(log_dir)

# -----------------------------

cv2.imwrite('img_human.png', view_human(env))
cv2.imwrite('img_state.png', view_machine(state))
cv2.imwrite('img_heatmap.png', attention_window(state, model, True))
cv2.imwrite('img_attention.png', attention_comp())
