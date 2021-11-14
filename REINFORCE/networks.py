#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop

def policy_gradient(input_shape, window_length, action_space, lr):
    inputs = layers.Input(shape=(input_shape[0], input_shape[1], window_length))

    layer1 = layers.Flatten()(inputs)
    layer2 = layers.Dense(512, activation="elu", kernel_initializer='he_uniform')(layer1)

    action = layers.Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(layer2)

    model = keras.Model(inputs=inputs, outputs=action)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))
    return model
