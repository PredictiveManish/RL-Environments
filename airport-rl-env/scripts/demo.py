#!/usr/bin/env python3
"""Airport demo: watch the airport scene and the ground vehicles driving.

By default this opens the 3D window and keeps it open until you close it,
pacing the simulation to real time so it is watchable. The vehicle is driven by
a simple hand-written "scripted" driver that steers toward a gate and dodges
obstacles using the lidar - so it looks alive without any training. Pass
``--policy random`` to use random actions instead.

Examples
--------
    # Watch it (window stays open until you close it)
    python scripts/demo.py --render human

    # Same, but only 3 episodes, random actions
    python scripts/demo.py --render human --no-loop --episodes 3 --policy random

    # Headless: save a short GIF
    python scripts/demo.py --render rgb_array --episodes 1 --save-gif airport.gif

    # Just measure the environment, no rendering
    python scripts/demo.py --render none --episodes 20
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import math
import time

import numpy as np

import gymnasium as gym
import airport_rl  # noqa: F401  (registers AirportGround-v0)


def scripted_action(raw) -> np.ndarray:
    """A tiny goal-seeking + lidar-avoidance controller (no learning).

    Works entirely in the vehicle's own frame: the goal provides an attractive
    direction, lidar hits push away from obstacles.
    """
    goal = raw._goal
    dx = float(goal[0]) - raw._x()
    dy = float(goal[1]) - raw._y()
    dist = math.hypot(dx, dy)
    yaw = raw._yaw()
    bearing = (math.atan2(dy, dx) - yaw + math.pi) % (2.0 * math.pi) - math.pi

    # Attraction toward the goal, expressed in the vehicle frame.
    ax, ay = math.cos(bearing), math.sin(bearing)

    # Repulsion from nearby lidar hits (rays are already in the vehicle frame).
    lidar = raw._lidar()
    n = len(lidar)
    rx = ry = 0.0
    thresh = 0.30
    for i, r in enumerate(lidar):
        if r < thresh:
            a = -math.pi + i * (2.0 * math.pi / n)
            w = (thresh - r) / thresh
            rx -= w * math.cos(a)
            ry -= w * math.sin(a)

    vx = ax + 1.6 * rx
    vy = ay + 1.6 * ry
    desired = math.atan2(vy, vx)  # angle to steer toward, in vehicle frame

    steer = float(np.clip(desired * 1.6, -1.0, 1.0))
    throttle = 1.0 if abs(desired) < 0.6 else 0.35
    if dist < raw.cfg.goal_radius * 1.5:
        throttle = min(throttle, 0.6)
    return np.array([throttle, steer], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Airport RL demo")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", choices=["human", "rgb_array", "none"], default="human")
    ap.add_argument("--policy", choices=["scripted", "random"], default="scripted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--fps", type=int, default=20, help="real-time playback rate for --render human")
    ap.add_argument("--save-gif", type=str, default="")
    ap.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True,
                    help="keep the window open until you close it (human render only)")
    args = ap.parse_args()

    render_mode = None if args.render == "none" else args.render
    env = gym.make("AirportGround-v0", render_mode=render_mode)
    raw = env.unwrapped
    frame_dt = 1.0 / max(args.fps, 1)

    frames = []
    successes = 0
    episode = 0
    window_closed = False

    while True:
        obs, info = env.reset(seed=args.seed + episode)
        ep_reward = 0.0
        for step in range(args.max_steps):
            if args.policy == "scripted":
                action = scripted_action(raw)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if render_mode == "human":
                env.render()
                # Stop if the user closed the window
                viewer = getattr(raw, "_viewer", None)
                if viewer is not None and not viewer.is_running():
                    window_closed = True
                    break
                time.sleep(frame_dt)  # pace to real time
            elif render_mode == "rgb_array":
                frames.append(env.render())

            if terminated or truncated:
                break

        if window_closed:
            print("window closed - exiting")
            break

        successes += int(info.get("is_success", False))
        print(
            f"episode {episode+1}: steps={step+1} reward={ep_reward:8.2f} "
            f"final_dist={info.get('distance', float('nan')):.1f} "
            f"success={info.get('is_success')} collision={info.get('collision')}"
        )
        episode += 1

        keep_looping = args.loop and render_mode == "human"
        if not keep_looping and episode >= args.episodes:
            break

    if episode:
        print(f"\n{successes}/{episode} episodes reached the gate ({args.policy} policy).")
    env.close()

    if args.save_gif and frames:
        _save_gif(frames, args.save_gif, args.fps)


def _save_gif(frames, path: str, fps: int) -> None:
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=fps)
        print(f"saved {len(frames)} frames to {path}")
    except Exception as exc:  # pragma: no cover
        print(f"could not save gif ({exc}); install imageio with: pip install imageio")


if __name__ == "__main__":
    main()
