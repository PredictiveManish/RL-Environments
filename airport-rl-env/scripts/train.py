#!/usr/bin/env python3
"""Train a PPO policy to drive a ground vehicle across the airport apron.

Requires stable-baselines3 (and torch)::

    pip install stable-baselines3

Then::

    python scripts/train.py --timesteps 300000 --n-envs 8 --out models/ppo_airport
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import os

import airport_rl  # noqa: F401  (registers AirportGround-v0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PPO on AirportGround-v0")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, default="models/ppo_airport")
    ap.add_argument("--logdir", type=str, default="runs/airport")
    ap.add_argument("--task", choices=["navigate", "dock_aircraft", "shuttle", "patrol"],
                    default="navigate", help="mission to train on")
    args = ap.parse_args()

    # Imported here so the rest of the project works without SB3 installed.
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Tensorboard logging is optional: fall back gracefully if it is absent.
    import importlib.util

    if importlib.util.find_spec("tensorboard") is not None:
        tb_log = args.logdir
        os.makedirs(args.logdir, exist_ok=True)
    else:
        tb_log = None
        print("note: tensorboard not installed - logging disabled "
              "(pip install tensorboard to enable it)")

    env_kwargs = {"task": args.task}
    env = make_vec_env("AirportGround-v0", n_envs=args.n_envs, seed=args.seed,
                       env_kwargs=env_kwargs)
    eval_env = make_vec_env("AirportGround-v0", n_envs=1, seed=args.seed + 100,
                            env_kwargs=env_kwargs)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log=tb_log,
        seed=args.seed,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(os.path.dirname(args.out) or ".", "best"),
        log_path=tb_log,
        eval_freq=max(5000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=False)
    model.save(args.out)
    print(f"saved model to {args.out}.zip")
    if tb_log:
        print(f"tensorboard:  tensorboard --logdir {args.logdir}")


if __name__ == "__main__":
    main()
