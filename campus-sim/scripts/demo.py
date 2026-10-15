"""
demo.py -- build a self-contained 3D viewer showing the campus and a set of
rollouts (the expert baseline + a random agent for contrast).

    python scripts/demo.py                # writes viewer/campus_viewer.html
    python scripts/demo.py --out my.html --n 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from campus_sim import CampusNavEnv, load_campus, build_task_sets           # noqa: E402
from campus_sim.policies import expert_action, random_action                # noqa: E402
from campus_sim.export import run_rollout, write_viewer                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "viewer", "campus_viewer.html"))
    ap.add_argument("--n", type=int, default=4, help="number of expert rollouts")
    ap.add_argument("--disruptions", type=int, default=1)
    args = ap.parse_args()

    campus = load_campus()
    tasks = build_task_sets(campus)
    env = CampusNavEnv(campus=campus, tasks=tasks["eval"], n_disruptions=args.disruptions)

    rollouts = []
    for k in range(args.n):
        r = run_rollout(env, expert_action, task_id=k, seed=100 + k,
                        label=f"expert #{k+1}")
        rollouts.append(r)
    # PPO rollouts if a trained model is present
    ppo_path = os.path.join(ROOT, "data", "ppo_campus.zip")
    if os.path.exists(ppo_path):
        try:
            from stable_baselines3 import PPO
            model = PPO.load(ppo_path)
            ppo_policy = lambda e: model.predict(e._obs(), deterministic=True)[0]
            for k in range(args.n):
                rollouts.append(run_rollout(env, ppo_policy, task_id=k, seed=100 + k,
                                            label=f"PPO #{k+1}"))
        except Exception as e:  # torch missing etc.
            print("  (skipping PPO rollouts:", e, ")")
    # one random-agent rollout for contrast
    rollouts.append(run_rollout(env, random_action, task_id=0, seed=7, label="random"))

    with open(os.path.join(ROOT, "data", "campus.json")) as f:
        campus_raw = json.load(f)

    write_viewer(args.out, campus_raw, rollouts,
                 title="IKGPTU Main Campus, Kapurthala \u2014 RL Environment")
    print(f"wrote {args.out}  ({len(rollouts)} rollouts)")
    for r in rollouts:
        print(f"  {r['label']:10s} task {r['task_id']:3d}  "
              f"{r['start_name'][:26]:26s} -> {r['goal_name'][:26]:26s}  "
              f"steps={r['steps']:3d} reward={r['cum_reward']:7.1f} "
              f"success={r['success']} collisions={r['n_collisions']}")


if __name__ == "__main__":
    main()
