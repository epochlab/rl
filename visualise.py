#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import cv2, skimage
from skimage import transform

from wrappers.doom import Sandbox
from networks import dqn, dueling_dqn
from utils import load_config, render_gif, load

# -----------------------------

config = load_config()['doom-ddqn']
log_dir = 'metrics/20210816-233347/'

dim = (640, 480)

# -----------------------------

def filter_summary(model):
    for layer in model.layers:
        if 'conv' in layer.name:
            filters, biases = layer.get_weights()
            print(layer.name, filters.shape)

def view_human(env):
    state = env.get_state()
    frame = state.screen_buffer
    frame = np.rollaxis(frame, 0, 3)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def view_depth(env):
    state = env.get_state()
    depth = state.depth_buffer
    return depth

def view_automap(env):
    state = env.get_state()
    automap = state.automap_buffer
    return automap

def view_labels(env):
    state = env.get_state()
    labels = state.labels
    return labels

def view_machine(state, factor):
    state = np.array(state)
    state = cv2.resize(state, (state.shape[0]*factor, state.shape[1]*factor))

    x0 = np.repeat(state[:, :, 0, np.newaxis], 3, axis=2)
    x1 = np.repeat(state[:, :, 1, np.newaxis], 3, axis=2)
    x2 = np.repeat(state[:, :, 2, np.newaxis], 3, axis=2)
    x3 = np.repeat(state[:, :, 3, np.newaxis], 3, axis=2)

    grid = np.concatenate((x0, x1, x2, x3), axis=1) * 255.0
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

def attention_comp(state):
    human = view_human(env)
    attention = attention_window(state, model, False)

    mask = np.zeros_like(human)
    mask[:,:,0] = attention
    mask[:,:,1] = attention
    mask[:,:,2] = attention

    comp = cv2.cvtColor(human, cv2.COLOR_RGB2BGR) * (mask / 255.0)
    return comp

def feature_model(model, depth):
    out = model.layers[depth].output
    feature_model = tf.keras.Model(inputs = model.inputs, outputs = out)
    return feature_model

def witness(env, action_space, model):
    info, prev_info, stack, state = sandbox.reset(env)
    frame_count = 0

    human_buf = []
    state_buf = []
    heatmap_buf = []
    attention_buf = []

    while not env.is_episode_finished():

        human_buf.append(cv2.cvtColor(view_human(env), cv2.COLOR_RGB2BGR))
        state_buf.append(view_machine(state, 2))
        heatmap_buf.append(attention_window(state, model, True))
        attention_buf.append(attention_comp(state))

        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()

        state_next, reward, terminal, info = sandbox.step(env, stack, prev_info, action, action_space)

        prev_info = info
        state = state_next
        frame_count += 1

        if terminal:
            break

    render_gif(human_buf, log_dir + "viz_human")
    render_gif(state_buf, log_dir + "viz_state")
    render_gif(heatmap_buf, log_dir + "viz_heatmap")
    render_gif(attention_buf, log_dir + "viz_attention")

# -----------------------------

sandbox = Sandbox(config)

env, action_space = sandbox.build_env(config['env_name'])
info, prev_info, stack, state = sandbox.reset(env)

model = load(log_dir)

# -----------------------------

witness(env, action_space, model)

# cv2.imwrite('img_human.png', view_human(env))
# cv2.imwrite('img_state.png', view_machine(state))
# cv2.imwrite('img_heatmap.png', attention_window(state, model, True))
# cv2.imwrite('img_attention.png', attention_comp())
