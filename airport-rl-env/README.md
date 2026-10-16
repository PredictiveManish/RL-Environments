# Airport RL — a real-world 3D airport as a Reinforcement-Learning environment

A ready-to-run **Gymnasium + MuJoCo** environment that depicts an airport
ground-operations scene and lets you train an RL agent to run ground vehicles on
the apron. It is the airport analogue of the kind of 3D multi-vehicle simulation
you may have seen for a port: a full 3D scene, several vehicles and aircraft
moving at once, and a standard RL interface you can plug a policy into.

The scene is a stylised, compressed model of a single-runway international
airport (proportions loosely following Noida International "Jewar" Airport):

- a **runway** with markings and a **parallel taxiway** carrying **live taxiing aircraft**,
- an **apron** with **six gate stands**, each with a parked aircraft and turnaround trucks,
- ambient **ground traffic**, a **control tower**, **baggage hall**, **fuel farm** and **maintenance hangar**.

![3/4 view of the airport](docs/airport_hero.png)
![Top-down view of the airport](docs/airport_topdown.png)

---

## Missions

The agent is a **ground vehicle** (baggage tug / pushback tractor). Assign it a
job with `task=`. Every mission is a route of one or more waypoints; the episode
ends on success (all waypoints reached), on collision, or on timeout.

| `task` | The vehicle must… |
|---|---|
| `navigate` | drive to a gate stand (point-to-point) |
| `dock_aircraft` | dock precisely at the nose of a parked aircraft (tight tolerance) |
| `shuttle` | run a two-leg delivery: baggage hall → gate stand |
| `patrol` | follow a multi-waypoint route along the apron lanes |

| | |
|---|---|
| **Observation** | `Box(30,)`: ego speed, heading (sin/cos), distance & bearing to the current waypoint, plus 24 lidar range readings. Optional RGB image. |
| **Action** | `Box(2,)` = `[throttle, steering]`, each in `[-1, 1]` |
| **Reward** | progress toward the current waypoint + heading bonus − step cost + a bonus per intermediate waypoint; **+50** on completing the mission, **−50** (and termination) on collision |
| **Ends** | mission complete, collision, or 800 steps |

---

## Quick start

```bash
# 1. install (a single command — no C++ compiler needed)
pip install -r requirements.txt

# 2. watch it running with the built-in scripted driver
python scripts/demo.py --render human                       # navigate (default)
python scripts/demo.py --render human --task shuttle        # try another mission
python scripts/demo.py --render human --task patrol

# 3. render preview stills to PNG
python scripts/preview.py --outdir previews

# 4. train a PPO policy on a chosen mission
python scripts/train.py --timesteps 1000000 --n-envs 16 --task navigate --out models/ppo_airport

# 5. watch the trained policy
python scripts/enjoy.py --model models/ppo_airport --render human
```

No display (server / CI)? Use headless mode:

```bash
MUJOCO_GL=egl python scripts/demo.py --render none --episodes 20 --task patrol
MUJOCO_GL=egl python scripts/demo.py --render rgb_array --episodes 1 --save-gif out.gif
```

Run the tests:

```bash
pip install pytest
pytest -q
```

---

## Project structure

```
airport-rl-env/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── docs/                        # preview images
├── airport_rl/                  # the installable package
│   ├── __init__.py              # registers "AirportGround-v0"
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── airport_env.py       # ← THE ENVIRONMENT (Gymnasium + MuJoCo)
│   │   └── config.py            # ← airport geometry, missions, RL hyper-parameters
│   ├── scene/
│   │   ├── __init__.py
│   │   ├── airport_builder.py   # ← builds the whole airport as MJCF
│   │   ├── vehicle_factory.py   # agent, traffic, turnaround trucks
│   │   └── aircraft_factory.py  # parked + taxiing aircraft
│   └── utils/
│       ├── __init__.py
│       └── math_utils.py
├── scripts/
│   ├── demo.py                  # scripted / random driver, with --task
│   ├── preview.py               # render stills
│   ├── train.py                 # PPO training (--task)
│   └── enjoy.py                 # run a trained model
└── tests/
    └── test_env.py
```

**Where to edit what**

- Airport geometry (runway, gates, taxi aircraft, turnaround trucks, colours) →
  `airport_rl/envs/config.py` (`AirportConfig`).
- Missions and RL problem (task list, tolerances, speed, reward, lidar, episode
  length) → `airport_rl/envs/config.py` (`EnvConfig`).
- How the scene is assembled → `airport_rl/scene/airport_builder.py`.
- Vehicle / aircraft shapes → `vehicle_factory.py` / `aircraft_factory.py`.

---

## Using it from your own code

```python
import gymnasium as gym
import airport_rl                       # registers "AirportGround-v0"

env = gym.make("AirportGround-v0", render_mode="human", task="shuttle")
obs, info = env.reset(seed=0)
for _ in range(2000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

You can also change the mission per episode: `env.reset(options={"task": "patrol"})`.

Why MuJoCo rather than PyBullet? PyBullet ships **no prebuilt wheels** and
compiles from source, which needs a C++ toolchain (a common blocker on Windows).
MuJoCo installs from a single `pip install` on Windows/macOS/Linux, renders well,
and is the current standard for RL environments.

---

## Dropping in real open-source 3D models

The scene is built entirely from primitives so it runs with zero downloads. If
you want photorealistic assets (e.g. CC0 models of an A320, a terminal, a tug
from Poly Haven / Sketchfab / Free3D), MuJoCo can load meshes directly:

1. Put the mesh under `airport_rl/assets/`, e.g. `airport_rl/assets/a320.obj`.
2. Reference it in the MJCF built by `airport_builder.py`:

   ```xml
   <asset>
     <mesh name="a320" file="airport_rl/assets/a320.obj" scale="0.01 0.01 0.01"/>
   </asset>
   <worldbody>
     <body name="aircraft_0" pos="0 -8 2.6">
       <geom name="aircraft_0_body" type="mesh" mesh="a320" rgba="0.9 0.92 0.95 1"/>
     </body>
   </worldbody>
   ```

3. Keep a simple box/capsule as the **collision** geom (add `contype`/`conaffinity`)
   and set the mesh geom to `contype="0" conaffinity="0"` if you only want it for
   looks. Lidar rays and collisions keep working off the collision geom.

`scale` is important — most downloaded models are in centimetres or inches.

---

## Troubleshooting

- **`ImportError: No module named 'airport_rl'`** — either `pip install -e .`
  from the project root, or run the scripts from the project root (the scripts
  add the root to `sys.path` themselves).
- **Blank/black renders on a headless machine** — set `MUJOCO_GL=egl`
  (or `osmesa`) before the command. On a normal desktop no variable is needed.
- **Training is slow on CPU** — reduce `--n-envs`, or install a CUDA build of
  torch for GPU training.
- **A mission feels too hard** — longer routes (`shuttle`, `patrol`) need the
  full 800-step budget; you can raise `max_steps` or lower `max_speed` in
  `EnvConfig` to make them easier.

---

## License / credits

Project code: MIT. The airport is a stylised original model generated from
primitives — it is not survey data of any real airport.
