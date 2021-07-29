#!/usr/bin/env python3

import vizdoom

class sandbox:
    def __init__(self):
        self.CONFIG_PATH = '/mnt/vanguard/git/ViZDoom-master/scenarios/basic.cfg'
        self.MAP = 'map01'

    def build_env(self):
        INPUT_SHAPE = (64, 64)
        WINDOW_LENGTH = 4

        env = vizdoom.DoomGame()
        env.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        env.set_window_visible(False)
        env.init()
        env.new_episode()
        return env, INPUT_SHAPE, WINDOW_LENGTH

    def interact(self, env):
        state = env.get_state()
        info = state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
        prev_info = info

        action_space = env.get_available_buttons_size()

        return state, info, prev_info, action_space
