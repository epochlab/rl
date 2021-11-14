#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop

def policy_gradient(input_shape, window_length, action_space, lr):
    inputs = layers.Input(shape=(input_shape[0], input_shape[1], window_length))

    layer1 = layers.Flatten()(inputs)
    layer2 = layers.Dense(512, activation="elu", kernel_initializer='he_uniform')(layer1)

    action = layers.Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(layer2)

    model = keras.Model(inputs=inputs, outputs=action)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))
    return model

def actor_critic(input_shape, window_length, action_space, lr):
    inputs = layers.Input(shape=(input_shape[0], input_shape[1], window_length))

    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="elu", kernel_initializer='he_uniform')(layer4)

    action = layers.Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(layer5)
    value = layers.Dense(1, kernel_initializer='he_uniform')(layer5)

    actor = keras.Model(inputs=inputs, outputs=action)
    actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))

    critic = keras.Model(inputs=inputs, outputs=value)
    critic.compile(loss='mse', optimizer=RMSprop(lr=lr))

    return actor, critic

def actor_critic_ppo(input_shape, window_length, action_space, lr):
    inputs = layers.Input(shape=(input_shape[0], input_shape[1], window_length))

    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="elu", kernel_initializer='he_uniform')(layer4)

    action = layers.Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(layer5)
    value = layers.Dense(1, kernel_initializer='he_uniform')(layer5)

    def ppo_loss(y_true, y_pred):
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+action_space], y_true[:, 1+action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 5e-3

        prob = y_pred * actions
        old_prob = actions * prediction_picks
        r = prob/(old_prob + 1e-10)
        p1 = r * advantages
        p2 = K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages
        loss =  -K.mean(K.minimum(p1, p2) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
        return loss

    actor = keras.Model(inputs=inputs, outputs=action)
    actor.compile(loss=ppo_loss, optimizer=RMSprop(lr=lr))

    critic = keras.Model(inputs=inputs, outputs=value)
    critic.compile(loss='mse', optimizer=RMSprop(lr=lr))

    return actor, critic
