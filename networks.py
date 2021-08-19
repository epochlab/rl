#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# DQN -------------------------

def dqn(input_shape, window_length, action_space):
    inputs = layers.Input(shape=(input_shape[0], input_shape[1], window_length,))

    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu", kernel_initializer='he_uniform')(layer4)

    action = layers.Dense(action_space, activation="linear", kernel_initializer='he_uniform')(layer5)
    return keras.Model(inputs=inputs, outputs=action)

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
    return keras.Model(inputs=inputs, outputs=action)

# A2C -------------------------

def actor_network(input_shape, action_space):
    inputs = layers.Input(shape=(input_shape))

    layer1 = layers.Conv2D(32, 8, strides=4)(inputs)
    layer1 = layers.BatchNormalization()(layer1)
    layer1 = layers.Activation('relu')(layer1)

    layer2 = layers.Conv2D(64, 4, strides=2)(layer1)
    layer2 = layers.BatchNormalization()(layer2)
    layer2 = layers.Activation('relu')(layer2)

    layer3 = layers.Conv2D(64, 3, strides=1)(layer2)
    layer3 = layers.BatchNormalization()(layer3)
    layer3 = layers.Activation('relu')(layer3)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(output_dim=64)(layer4)
    layer5 = layers.BatchNormalization()(layer5)
    layer5 = layers.Activation('relu')(layer5)

    layer6 = layers.Dense(output_dim=32)(layer5)
    layer6 = layers.BatchNormalization()(layer6)
    layer6 = layers.Activation('relu')(layer6)

    action = layers.Dense(output_dim=action_space, activation='softmax')(layer6)
    return keras.Model(inputs=inputs, outputs=action)

def critic_network(input_shape, action_space):
    inputs = layers.Input(shape=(input_shape))

    layer1 = layers.Conv2D(32, 8, strides=4)(inputs)
    layer1 = layers.BatchNormalization()(layer1)
    layer1 = layers.Activation('relu')(layer1)

    layer2 = layers.Conv2D(64, 4, strides=2)(layer1)
    layer2 = layers.BatchNormalization()(layer2)
    layer2 = layers.Activation('relu')(layer2)

    layer3 = layers.Conv2D(64, 3, strides=1)(layer2)
    layer3 = layers.BatchNormalization()(layer3)
    layer3 = layers.Activation('relu')(layer3)

    layer4 = layers.Flatten()(layer3)
    
    layer5 = layers.Dense(output_dim=64)(layer4)
    layer5 = layers.BatchNormalization()(layer5)
    layer5 = layers.Activation('relu')(layer5)

    layer6 = layers.Dense(output_dim=32)(layer5)
    layer6 = layers.BatchNormalization()(layer6)
    layer6 = layers.Activation('relu')(layer6)

    action = layers.Dense(output_dim=action_space, activation='linear')(layer6)
    return keras.Model(inputs=inputs, outputs=action)
