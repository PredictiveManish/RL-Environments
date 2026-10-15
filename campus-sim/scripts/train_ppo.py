"""
train_ppo.py -- train a real RL agent on the campus environment with PPO
(Stable-Baselines3). Requires: pip install stable-baselines3

    python scripts/train_ppo.py --steps 300000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from campus_sim import CampusNavEnv, load_campus, build_task_sets   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _has_tensorboard():
    try:
        import tensorboard  # noqa: F401
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--disruptions", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "ppo_campus"))
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy

    campus = load_campus()
    ts = build_task_sets(campus)

    def make(task_set):
        return lambda: CampusNavEnv(campus=campus, tasks=ts[task_set],
                                    n_disruptions=args.disruptions)

    train = make_vec_env(make("train"), n_envs=args.n_envs, seed=args.seed)
    tb = os.path.join(ROOT, "out", "tb") if _has_tensorboard() else None
    model = PPO("MlpPolicy", train, n_steps=256, batch_size=256, gamma=0.99,
                learning_rate=3e-4, ent_coef=0.005, verbose=1, seed=args.seed,
                tensorboard_log=tb)
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(args.out)
    print("saved", args.out + ".zip")

    eval_env = make("eval")()
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=50,
                                    deterministic=True)
    # success rate
    wins = 0
    for i in range(len(ts["eval"])):
        obs, _ = eval_env.reset(seed=0, options={"task_id": i})
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(a)
            done = term or trunc
        wins += int(info.get("success", False))
    print(f"eval mean return = {mean_r:.1f} +/- {std_r:.1f}")
    print(f"eval success rate = {wins}/{len(ts['eval'])} = {100*wins/len(ts['eval']):.0f}%")


if __name__ == "__main__":
    main()
