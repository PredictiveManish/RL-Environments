"""
train_cem.py -- dependency-free training demo.

Uses the Cross-Entropy Method over a small linear policy, so it needs only
numpy + gymnasium (no PyTorch). It will not beat a proper PPO run, but it is a
real learning loop that improves the agent and shows the environment trains.

    python scripts/train_cem.py --iters 25
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from campus_sim import CampusNavEnv, load_campus, build_task_sets   # noqa: E402

OBS, ACT = 8, 2


def unpack(theta):
    W = theta[:ACT * OBS].reshape(ACT, OBS)
    b = theta[ACT * OBS:]
    return W, b


def policy(theta):
    W, b = unpack(theta)

    def act(env):
        a = np.tanh(W @ env._obs() + b)
        return a.astype(np.float32)
    return act


def evaluate(env, theta, task_ids, max_steps=250):
    total = 0.0
    wins = 0
    for tid in task_ids:
        env.reset(seed=0, options={"task_id": int(tid)})
        done = False
        steps = 0
        while not done and steps < max_steps:
            _, r, term, trunc, info = env.step(policy(theta)(env))
            total += r
            done = term or trunc
            steps += 1
        wins += int(info.get("success", False))
    return total / len(task_ids), wins / len(task_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--tasks", type=int, default=6, help="train tasks per candidate")
    ap.add_argument("--disruptions", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    campus = load_campus()
    ts = build_task_sets(campus)
    env = CampusNavEnv(campus=campus, tasks=ts["train"], n_disruptions=args.disruptions)
    eval_env = CampusNavEnv(campus=campus, tasks=ts["eval"], n_disruptions=args.disruptions)

    dim = ACT * OBS + ACT
    mean = np.zeros(dim)
    std = np.ones(dim) * 0.6
    train_ids = rng.choice(len(ts["train"]), size=args.tasks, replace=False)

    t0 = time.time()
    for it in range(args.iters):
        pop = [mean + std * rng.standard_normal(dim) for _ in range(args.pop)]
        scores = [evaluate(env, th, train_ids)[0] for th in pop]
        order = np.argsort(scores)[::-1]
        elite = np.array([pop[i] for i in order[:args.elite]])
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-3
        if it % 5 == 0 or it == args.iters - 1:
            ev_ret, ev_win = evaluate(eval_env, mean, range(len(ts["eval"])))
            print(f"iter {it:3d}  best_train={scores[order[0]]:8.1f}  "
                  f"eval_return={ev_ret:8.1f}  eval_success={ev_win*100:4.0f}%  "
                  f"[{time.time()-t0:5.1f}s]", flush=True)

    np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "cem_policy.npy"), mean)
    print("saved data/cem_policy.npy")


if __name__ == "__main__":
    main()
