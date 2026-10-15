#!/usr/bin/env python3
"""Render the airport to PNG images (no training needed).

    python scripts/preview.py --outdir previews

Produces:
    previews/airport_topdown.png   - whole airport from above
    previews/airport_chase.png     - chase camera behind the vehicle

Note: offscreen rendering needs a working OpenGL context. On a normal desktop
this just works. On a headless server, set ``MUJOCO_GL=egl`` (or ``osmesa``).
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import os

import numpy as np
import mujoco

import airport_rl  # noqa: F401
from airport_rl.envs.airport_env import AirportGroundEnv


def _save(img: np.ndarray, path: str) -> None:
    try:
        from PIL import Image

        Image.fromarray(img).save(path)
    except Exception:
        import imageio.v2 as imageio

        imageio.imwrite(path, img)
    print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="previews")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    env = AirportGroundEnv(render_mode=None)
    env.reset(seed=0)

    renderer = mujoco.Renderer(env.model, args.height, args.width)

    # ---- Top-down view of the whole airport --------------------------
    top = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, top)
    top.lookat[:] = [0.0, -30.0, 0.0]
    top.distance = 320.0
    top.elevation = -89.0
    top.azimuth = 90.0
    renderer.update_scene(env.data, camera=top)
    _save(renderer.render().copy(), os.path.join(args.outdir, "airport_topdown.png"))

    # ---- 3/4 "hero" view of the whole airport -----------------------
    hero = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, hero)
    hero.lookat[:] = [0.0, -45.0, 0.0]
    hero.distance = 250.0
    hero.elevation = -28.0
    hero.azimuth = -70.0
    renderer.update_scene(env.data, camera=hero)
    _save(renderer.render().copy(), os.path.join(args.outdir, "airport_hero.png"))

    # ---- Chase camera behind the vehicle -----------------------------
    chase = env._make_camera()
    renderer.update_scene(env.data, camera=chase)
    _save(renderer.render().copy(), os.path.join(args.outdir, "airport_chase.png"))

    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
