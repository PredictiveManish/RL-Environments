"""
random_rollout.py -- the smallest possible example: drive the environment with
random actions and print what happens. Good first thing to run.

    python scripts/random_rollout.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from campus_sim import CampusNavEnv, load_campus, build_task_sets   # noqa: E402

env = CampusNavEnv(campus=load_campus(), tasks=build_task_sets(load_campus())["train"],
                   n_disruptions=1)
obs, info = env.reset(seed=0, options={"task_id": 0})
print("task:", info["start_name"], "->", info["goal_name"])

total = 0.0
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    total += reward
    done = terminated or truncated

print(f"steps={env.steps}  total_reward={total:.1f}  "
      f"success={info['success']}  collisions={env.n_collisions}")
print("tip: run  python scripts/demo.py  to watch it in 3D")
