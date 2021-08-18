#!/usr/bin/env python3

import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K

from agent import Agent
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
    human = sandbox.view_human(env)
    attention = attention_window(state, model, False)

    mask = np.zeros_like(human)
    mask[:,:,0] = attention
    mask[:,:,1] = attention
    mask[:,:,2] = attention

    comp = human * (mask / 255.0)
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

        human_buf.append(sandbox.view_human(env))
        state_buf.append(view_machine(state, 2))
        heatmap_buf.append(attention_window(state, model, True))
        attention_buf.append(attention_comp(state))

        action = agent.get_action(state, model)
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

agent = Agent(config, sandbox, env, action_space)

model = load(log_dir)

# -----------------------------

witness(env, action_space, model)
