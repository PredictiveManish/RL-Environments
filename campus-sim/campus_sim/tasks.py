"""Seeded task generation -- the 'work orders' the agent is asked to complete.

Mirrors the port environment's design: a large fixed training set plus a small
held-out evaluation set, both fully determined by a seed so runs are reproducible.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import Campus


def _poi_xy(campus: Campus):
    return [(p["name"], p["xy"][0], p["xy"][1]) for p in campus.pois]


def generate_tasks(campus: Campus, n: int, seed: int, min_sep: float = 120.0):
    """Return a list of tasks, each: {id, start, goal, start_name, goal_name}."""
    pois = _poi_xy(campus)
    rng = np.random.default_rng(seed)
    tasks = []
    tries = 0
    while len(tasks) < n and tries < n * 200:
        tries += 1
        i, j = rng.choice(len(pois), size=2, replace=False)
        sn, sx, sy = pois[i]
        gn, gx, gy = pois[j]
        if math.hypot(gx - sx, gy - sy) < min_sep:
            continue
        # keep the straight-line start->goal path from starting inside a building
        if campus.collides(sx, sy, 0.0) or campus.collides(gx, gy, 0.0):
            continue
        tasks.append({
            "id": len(tasks),
            "start": [float(sx), float(sy)],
            "goal": [float(gx), float(gy)],
            "start_name": sn,
            "goal_name": gn,
        })
    return tasks


TRAIN_SEED = 20261015
EVAL_SEED = 20261016
N_TRAIN = 1000
N_EVAL = 50


def build_task_sets(campus: Campus):
    return {
        "train": generate_tasks(campus, N_TRAIN, TRAIN_SEED),
        "eval": generate_tasks(campus, N_EVAL, EVAL_SEED),
    }
