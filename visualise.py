#!/usr/bin/env python3

import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from wrappers.doom import Sandbox
from agent import Agent
from networks import dqn, dueling_dqn
from utils import load_config, render_gif, load

# -----------------------------

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("GPU is", "available" if physical_devices else "NOT AVAILABLE")
print("Eager mode:", tf.executing_eagerly())

# -----------------------------

config = load_config('config.yml')['doom-dqn']
log_dir = 'metrics/20210820-222647/'

dim = (640, 480)

# -----------------------------

def filter_summary(model):
    for layer in model.layers:
        print(layer.name)

def intermediate_representation(state, model, layer_names=None):
    if isinstance(layer_names, list) or isinstance(layer_names, tuple):
        layers = [model.get_layer(name=layer_name).output for layer_name in layer_names]
    else:
        layers = model.get_layer(name=layer_names).output

    temp_model = tf.keras.Model(model.inputs, layers)
    prediction = temp_model.predict(state[np.newaxis, :, :, :])
    return prediction

def view_machine(state, factor):
    state = np.array(state)
    state = cv2.resize(state, (state.shape[0]*factor, state.shape[1]*factor))

    x0 = np.repeat(state[:, :, 0, np.newaxis], 3, axis=2)
    x1 = np.repeat(state[:, :, 1, np.newaxis], 3, axis=2)
    x2 = np.repeat(state[:, :, 2, np.newaxis], 3, axis=2)
    x3 = np.repeat(state[:, :, 3, np.newaxis], 3, axis=2)

    grid = np.concatenate((x0, x1, x2, x3), axis=1) * 255.0
    return grid

def attention_window(frame, model):
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
        atten_map = np.expand_dims(atten_map, axis=0)
        return atten_map

def attention_comp(state):
    human = sandbox.view_human(env)
    attention = attention_window(state, model)

    mask = np.zeros_like(human)
    mask[:,:,0] = attention
    mask[:,:,1] = attention
    mask[:,:,2] = attention

    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_TURBO)

    comp = human * (mask / 255.0)
    return heatmap, comp

def plot_value(values, counter, depth):
    s = np.array(counter)[-depth:]
    v = np.array(values)[-depth:]

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.plot(s, v)
    ax.grid()
    ax.set(xlabel='Time (s)', ylabel='Q-Value (V[s])', title='Temporal estimation of q-values.')

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return frame

def witness(env, action_space, model):
    print("Witnessing...", log_dir)
    info, prev_info, stack, state = sandbox.reset(env)
    frame_count = 0

    human_buf = []
    state_buf = []
    depth_buf = []
    automap_buf = []

    heatmap_buf = []
    attention_buf = []

    graph_buf = []
    values = []
    counter = []

    while not env.is_episode_finished():

        human_buf.append(sandbox.view_human(env))
        state_buf.append(view_machine(state, 2))
        depth_buf.append(sandbox.view_depth(env))
        automap_buf.append(sandbox.view_automap(env))

        heatmap, comp = attention_comp(state)
        heatmap_buf.append(heatmap)
        attention_buf.append(comp)

        q_val, action_prob = intermediate_representation(state, model, ['lambda', 'add'])
        action = tf.argmax(action_prob[0]).numpy()

        print("Frame:", frame_count, "| Q Value:", q_val[0], "| Action:", action)

        values.append(float(q_val[0]))
        counter.append(frame_count)
        graph = plot_value(values, counter, 50)
        graph_buf.append(graph)

        state_next, reward, terminal, info = sandbox.step(env, stack, prev_info, action, action_space)

        prev_info = info
        state = state_next
        frame_count += 1

        if terminal:
            break

    render_gif(human_buf, log_dir + 'viz_human')
    render_gif(state_buf, log_dir + 'viz_state')
    render_gif(depth_buf, log_dir + 'viz_depth')
    render_gif(automap_buf, log_dir + 'viz_automap')

    render_gif(heatmap_buf, log_dir + 'viz_heatmap')
    render_gif(attention_buf, log_dir + 'viz_attention')
    render_gif(graph_buf, log_dir + 'viz_graph')

# -----------------------------

sandbox = Sandbox(config)

env, action_space = sandbox.build_env(config['env_name'], True)
info, prev_info, stack, state = sandbox.reset(env)

agent = Agent(config, sandbox, env, action_space)
model = load(log_dir)

# -----------------------------

witness(env, action_space, model)
