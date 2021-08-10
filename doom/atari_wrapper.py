#!/usr/bin/env python3

from baselines.common.atari_wrappers import make_atari, wrap_deepmind

def build_env(env_name):
    env = make_atari(env_name)
    env = wrap_deepmind(env, frame_stack=True, scale=True)                      # Warp the frames, grey scale, stake four frame and scale to smaller ratio
    env.seed(1)

    print("Launching: Atari-5600 ||", env_name)

    screen_space = env.observation_space
    num_states = env.observation_space.shape
    action_space = env.action_space.n

    print('Frame:', screen_space)
    print('States:', num_states[0])
    print('Actions:', action_space)

    env.unwrapped.get_action_meanings()
    return env, action_space
