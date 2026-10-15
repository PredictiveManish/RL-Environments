#!/usr/bin/env python3
"""Watch a trained policy drive the vehicle, or run a saved model headless.

    python scripts/enjoy.py --model models/ppo_airport --render human
    python scripts/enjoy.py --model models/ppo_airport --render none --episodes 20
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse

import numpy as np

import gymnasium as gym
import airport_rl  # noqa: F401  (registers AirportGround-v0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a trained Airport RL policy")
    ap.add_argument("--model", type=str, default="models/ppo_airport")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", choices=["human", "rgb_array", "none"], default="human")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--save-gif", type=str, default="")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(args.model, device="auto")

    render_mode = None if args.render == "none" else args.render
    env = gym.make("AirportGround-v0", render_mode=render_mode)

    frames = []
    successes = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if render_mode == "human":
                env.render()
            elif render_mode == "rgb_array":
                frames.append(env.render())
            if terminated or truncated:
                break
        successes += int(info.get("is_success", False))
        print(
            f"episode {ep+1}: steps={step+1} reward={ep_reward:8.2f} "
            f"success={info.get('is_success')} collision={info.get('collision')}"
        )
    print(f"\n{successes}/{args.episodes} episodes reached the gate.")
    env.close()

    if args.save_gif and frames:
        try:
            import imageio.v2 as imageio

            imageio.mimsave(args.save_gif, frames, fps=args.fps)
            print(f"saved gif to {args.save_gif}")
        except Exception as exc:  # pragma: no cover
            print(f"could not save gif ({exc}); pip install imageio")


if __name__ == "__main__":
    main()
