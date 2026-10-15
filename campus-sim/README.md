# CampusSim — IKGPTU Main Campus, Kapurthala

A reinforcement-learning environment that rebuilds a real university campus as a
deterministic simulation, with a 3D viewer for every rollout — built to mirror
the *"real-world work turned into an RL simulation, every rollout rendered in
3D"* pattern of Adithya S K's **PortSimEnv** (port berth planning, Port of
Barcelona).

Here the "real world work" is: **an autonomous campus shuttle driving between
the buildings of I. K. Gujral Punjab Technical University, Main Campus,
Kapurthala.**

```
start (Boys Hostel)  ───────────────────────────────►  goal (Guest House)
        [campus.json: real boundary + buildings]     [reward: reach it, don't crash]
```

---

## What you get

| Piece | What it is |
|---|---|
| `campus_sim/` | The Gymnasium environment (`IKGPTU-CampusNav-v0`) |
| `data/campus.json` | The world model — campus boundary, buildings, roads, entrances |
| `viewer/campus_viewer.html` | **Open this file in a browser.** Self-contained 3D viewer that animates every rollout. No server, no internet needed. |
| `scripts/` | Build the campus, run a rollout, make the viewer, train with PPO or CEM, refresh real footprints from OSM |
| `SPEC.md` | A written spec of the environment — task, observation, action, reward, termination, determinism. |

---

## Quickstart

```bash
cd campus-sim
pip install -r requirements.txt

# 1. regenerate the world model (already built and included)
python scripts/build_campus.py

# 2. drive it with random actions — the smallest possible example
python scripts/random_rollout.py

# 3. build the 3D viewer and open viewer/campus_viewer.html in your browser
python scripts/demo.py
```

Then **double-click `viewer/campus_viewer.html`**. You get the campus in 3D, and
you can step through the expert baseline, a trained PPO agent, and a random
agent side by side, with a HUD showing reward, distance, collisions and status.

### Train an agent

```bash
# real RL, needs PyTorch  (pip install stable-baselines3)
python scripts/train_ppo.py --steps 300000

# dependency-free learning loop (numpy only, weaker — a smoke test)
python scripts/train_cem.py --iters 40
```

Verified numbers on the 50 held-out evaluation tasks (this machine, ~150k PPO
steps): **random agent 0% · CEM baseline ~12% · PPO ~36% · hand-coded expert
100%**. More PPO steps push it higher; the expert shows the task is solvable.

---

## Use it from Python

```python
from campus_sim import CampusNavEnv, load_campus, build_task_sets

env = CampusNavEnv(campus=load_campus(),
                   tasks=build_task_sets(load_campus())["train"],
                   n_disruptions=1)          # 0 = clean, 1+ = closed zones on route

obs, info = env.reset(seed=0, options={"task_id": 0})
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
print(info["success"], info["distance"])
```

Standard Gymnasium: `reset() -> (obs, info)`, `step(a) -> (obs, reward, terminated, truncated, info)`.

---

## Project layout

```
campus-sim/
├── campus_sim/
│   ├── geometry.py      # polygons, point-in-polygon, collision (numpy-vectorised)
│   ├── campus_env.py    # the Gymnasium environment  <-- the core
│   ├── tasks.py         # seeded task generation (1000 train + 50 eval)
│   ├── policies.py      # expert + random baselines
│   └── export.py        # rollout capture + self-contained 3D viewer builder
├── data/
│   ├── campus.json      # the world model
│   └── ppo_campus.zip   # a trained PPO model (optional)
├── viewer/
│   ├── campus_viewer.html   # the 3D viewer (generated, self-contained)
│   └── vendor/              # three.js + OrbitControls (bundled, works offline)
└── scripts/
    ├── build_campus.py      # build data/campus.json
    ├── demo.py              # build the 3D viewer from rollouts
    ├── random_rollout.py    # hello-world
    ├── train_ppo.py         # Stable-Baselines3 PPO
    ├── train_cem.py         # numpy-only Cross-Entropy Method
    └── refresh_from_osm.py  # swap in REAL OSM building footprints
```

---

## Where the geometry comes from (read this)

Honesty matters more than a pretty map:

* **Campus boundary — REAL.** OpenStreetMap way `123246355`
  ("I. K. Gujral Punjab Technical University", `amenity=university`), fetched via
  the OSM Nominatim API. Its area works out to **~79 acres**, matching the
  university's stated **78.16-acre** main campus.
* **Auditorium — REAL.** "Sri Guru Nanak Dev Ji Auditorium" is a mapped OSM
  feature; its true coordinates anchor the layout.
* **Individual building footprints — ILLUSTRATIVE.** The campus buildings are
  not individually mapped in OpenStreetMap (only the auditorium is). The 12
  footprints in `campus.json` are a documented, plausible layout placed **inside
  the real boundary**, sized and named after the facilities the university
  lists publicly (central library / Knowledge Resource Centre, academic blocks,
  hostels, cafeteria, workshop, sports complex, guest house, main building).

To replace the illustrative footprints with **real OSM footprints**, run this on
any machine whose network can reach the Overpass API (the sandbox this was built
in could not):

```bash
python scripts/refresh_from_osm.py --apply
```

That rewrites `data/campus.json` with true footprints. **No environment code
changes** — the env only ever reads `campus.json`. This is the whole point of
separating the world model from the simulation logic.

Data © OpenStreetMap contributors, [ODbL](https://www.openstreetmap.org/copyright).

---

## Make it *your* campus (or port, or warehouse)

1. Replace `data/campus.json` — `boundary`, `buildings` (each with a `polygon`
   of local-metre `[x, y]` points and a `height`), `paths`, and `pois`.
   `scripts/refresh_from_osm.py` shows the OSM route; hand-editing works too.
2. Everything else (env, tasks, viewer, training) works unchanged.

Set `ORIGIN_LON` / `ORIGIN_LAT` in `scripts/build_campus.py` to your location and
the local-metre frame follows.

---

## Requirements

```
gymnasium>=0.29
numpy>=1.24
# optional, for real RL training:
stable-baselines3>=2.0
```

The 3D viewer needs nothing but a browser — three.js is bundled in
`viewer/vendor/`.

See **SPEC.md** for the full environment specification and the method for turning
any real-world operation into an RL environment.
