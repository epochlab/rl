#!/usr/bin/env python3

import vizdoom
import numpy as np

import skimage
from skimage import transform, color

class Sandbox:
    def __init__(self, config):
        self.INPUT_SHAPE = config['input_shape']
        self.WINDOW_LENGTH = config['window_length']
        self.FPS = config['fps']
        self.GRADE = config['grade']
        self.VISIBLE = config['visible']
        self.FACTOR = config['reward_factor']

        self.STACK = np.zeros((config['input_shape'][0], config['input_shape'][1], config['window_length']))

    def build_env(self, config_path, AOV=False):
        env = vizdoom.DoomGame()
        env.load_config(config_path)

        env.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        env.set_window_visible(self.VISIBLE)
        env.set_render_hud(False)

        if AOV:
            env.set_depth_buffer_enabled(True)
            env.set_labels_buffer_enabled(True)

            env.set_automap_buffer_enabled(True)
            env.set_automap_mode(vizdoom.AutomapMode.OBJECTS_WITH_SIZE)
            env.set_automap_rotate(False)

            env.add_available_game_variable(vizdoom.GameVariable.POSITION_X)
            env.add_available_game_variable(vizdoom.GameVariable.POSITION_Y)
            env.add_available_game_variable(vizdoom.GameVariable.POSITION_Z)

            env.add_game_args("+am_followplayer 1")
            env.add_game_args("+am_backcolor 000000")

        env.init()
        action_space = env.get_available_buttons_size()
        return env, action_space

    def preprocess(self, frame):
        frame = np.rollaxis(frame, 0, 3)
        frame = skimage.color.rgb2gray(frame)
        frame = skimage.transform.resize(frame, self.INPUT_SHAPE)
        frame = np.array(frame).astype(np.float32)
        return frame

    def framestack(self, state):
        frame = self.preprocess(state)
        self.STACK = np.roll(self.STACK, 1, axis=2)
        self.STACK[:,:,0] = frame
        return self.STACK

    def async_stack(self, state, image_memory):
        frame = self.preprocess(state)

        if image_memory.shape == (1,*self.STACK):
            image_memory = np.squeeze(image_memory)

        image_memory = np.roll(image_memory, 1, axis=2)
        image_memory[:,:,0] = frame
        return image_memory

    def reset(self, env):
        env.new_episode()
        terminal = False
        state = env.get_state()
        info = state.game_variables

        frame = state.screen_buffer
        for _ in range(self.WINDOW_LENGTH):
            state = self.framestack(frame)
        return terminal, state, info

    def async_reset(self, env):
        image_memory = self.STACK
        env.new_episode()
        terminal = False
        state = env.get_state()
        info = state.game_variables

        frame = state.screen_buffer
        for _ in range(self.WINDOW_LENGTH):
            state = self.async_stack(frame, image_memory)
        return terminal, state, info, image_memory

    def one_hot(self, env, action_idx):
        action = np.zeros([env.get_available_buttons_size()])
        action[action_idx] = 1
        return action

    def step(self, env, action_idx, prev_info):
        action = np.zeros([env.get_available_buttons_size()])
        action[action_idx] = 1
        action = action.astype(int)

        env.set_action(action.tolist())
        env.advance_action(self.FPS)

        state = env.get_state()
        terminal = env.is_episode_finished()
        reward = env.get_last_reward()

        if terminal:
            env.new_episode()
            state = env.get_state()
            next_frame = state.screen_buffer
            info = state.game_variables

        next_frame = state.screen_buffer
        next_state = self.framestack(next_frame)
        info = state.game_variables
        reward = self.shape_reward(reward, self.FACTOR, info, prev_info)
        return next_state, reward, terminal, info

    def async_step(self, env, action_idx, prev_info, image_memory):
        action = np.zeros([env.get_available_buttons_size()])
        action[action_idx] = 1
        action = action.astype(int)

        env.set_action(action.tolist())
        env.advance_action(self.FPS)

        state = env.get_state()
        terminal = env.is_episode_finished()
        reward = env.get_last_reward()

        if terminal:
            env.new_episode()
            state = env.get_state()
            next_frame = state.screen_buffer
            info = state.game_variables

        next_frame = state.screen_buffer
        next_state = self.async_stack(next_frame, image_memory)
        info = state.game_variables
        reward = self.shape_reward(reward, self.FACTOR, info, prev_info)
        return next_state, reward, terminal, info

    def shape_reward(self, reward, factor, info, prev_info):
        reward *= factor

        if (info[0] > prev_info[0]): # Kill count
            reward = reward + 1

        if (info[1] < prev_info[1]): # Ammo
            reward = reward - 0.1

        if (info[2] < prev_info[2]): # Health
            reward = reward - 0.1

        return reward

    def view_human(self, env):
        state = env.get_state()
        frame = state.screen_buffer
        frame = np.rollaxis(frame, 0, 3)
        return frame
