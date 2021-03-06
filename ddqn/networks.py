#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def dqn(input_shape, window_length, action_space):
    inputs = layers.Input(shape=(input_shape[0], input_shape[1], window_length,))

    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu", kernel_initializer='he_uniform')(layer4)

    action = layers.Dense(action_space, activation="linear", kernel_initializer='he_uniform')(layer5)

    model = keras.Model(inputs=inputs, outputs=action)
    model.compile(loss='mse')
    return model

def dueling_dqn(input_shape, window_length, action_space):
    inputs = layers.Input(shape=(input_shape[0], input_shape[1], window_length,))

    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu", kernel_initializer='he_uniform')(layer4)

    value = layers.Dense(1, kernel_initializer='he_uniform')(layer5)
    value = layers.Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(value)

    adv = layers.Dense(action_space, kernel_initializer='he_uniform')(layer5)
    adv = layers.Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(adv)

    action = layers.Add()([value, adv])

    model = keras.Model(inputs=inputs, outputs=action)
    model.compile(loss='mse')
    return model
