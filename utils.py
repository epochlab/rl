#!/usr/bin/env python3

import imageio

def capture(env, step, sequence):
    if step < 600:
        frame = env.render(mode='rgb_array')
        sequence.append(frame)
    return sequence

def render_gif(frames, filename):
    return imageio.mimsave(filename + '.gif', frames)
