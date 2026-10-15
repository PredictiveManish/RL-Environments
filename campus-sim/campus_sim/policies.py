"""Baseline policies. The expert is a simple steering heuristic -- it is not a
trained RL agent, it is the 'sanity baseline' that shows the task is solvable and
gives the 3D viewer something meaningful to replay."""
from __future__ import annotations

import math

import numpy as np

_ANGLES = [0, 15, -15, 30, -30, 50, -50, 75, -75, 100, -100, 130, -130, 180]


def _rot(v, deg):
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return np.array([v[0] * ca - v[1] * sa, v[0] * sa + v[1] * ca])


def expert_action(env):
    pos, goal = env.pos, env.goal
    to_goal = goal - pos
    d = float(np.linalg.norm(to_goal)) + 1e-9
    base = to_goal / d
    step = env.max_speed * env.dt
    best, best_score = None, -1e18
    for ang in _ANGLES:
        direction = _rot(base, ang)
        probe = pos + direction * step * 2.0
        if env._blocked(probe[0], probe[1]):
            continue
        score = (d - float(np.linalg.norm(goal - probe))) - abs(ang) * 0.03
        if score > best_score:
            best_score, best = score, direction
    if best is None:
        best = -base
    return np.clip(best, -1.0, 1.0).astype(np.float32)


def random_action(env):
    return env.action_space.sample()
