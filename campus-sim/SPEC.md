# SPEC.md — How to specify an RL environment, and the spec for this one

You said you weren't sure how to write the *specs* for an RL environment. This
document is that spec, written out — and it doubles as a template you can copy
for any other campus, port, warehouse or factory.

---

## Part 1 — The method (this is the whole trick)

Every "real world turned into an RL environment" follows the same five steps.
Adithya S K's PortSimEnv does exactly this for port berth planning; this project
does it for a campus shuttle.

1. **Pick a real operation with a measurable outcome.** Not "a cool 3D scene" —
   a job that has a *success* and a *cost*. Here: *get a shuttle from building A
   to building B, quickly, without crashing.* For a port: *schedule vessels into
   berths to minimise waiting time.*
2. **Freeze the world into data.** Geometry, entities, constraints. Here:
   `data/campus.json` (boundary, buildings, roads, entrances). For a port:
   berths, cranes, the vessel arrival table.
3. **Define the interface**: what the agent sees (observation), what it can do
   (action), and what it is rewarded for (reward). Everything else follows.
4. **Make it deterministic and seeded.** Same seed → same task, same physics.
   This is what makes an environment *trainable* and *comparable* across agents.
   It is also why PortSimEnv ships a fixed task set.
5. **Make it verifiable and not gameable.** The reward must reflect the real
   outcome, not a proxy the agent can cheat. Success here is literally
   "distance to goal < success_radius", which cannot be faked.

Then you bolt a **3D viewer** on top. The viewer is *presentation only* — it
reads the recorded state of a rollout and draws it. It must never affect the
simulation. This separation is why the reference video can show "every model
rollout in 3D" without the 3D being part of the learning.

---

## Part 2 — The spec for `IKGPTU-CampusNav-v0`

### Identity
| Field | Value |
|---|---|
| Environment id | `IKGPTU-CampusNav-v0` |
| Domain | Continuous 2D navigation / mobile robotics |
| API | [Gymnasium](https://gymnasium.farama.org/) (`reset` / `step`) |
| World | IKGPTU Main Campus, Kapurthala (real boundary, see README) |
| Determinism | Fully seeded. `reset(seed=s, options={"task_id": k})` is reproducible. |

### Task
An autonomous campus shuttle starts at a campus point and must reach a goal
building's entrance. The straight-line route is usually blocked by buildings, so
the agent must steer around obstacles while heading to the goal.

### Observation — `Box(8,)`, roughly in `[-1, 1]`
| # | Meaning |
|---|---|
| 0,1 | agent position (x, y), scaled by the campus half-diagonal `S` |
| 2,3 | goal position (x, y), scaled by `S` |
| 4,5 | agent velocity (vx, vy), scaled by max speed |
| 6,7 | unit bearing to the goal: `(sin θ, cos θ)` |

`S` and max speed are exposed on the env. *Extension:* append K lidar ray
distances to the nearest obstacle for a more learnable observation.

### Action — `Box(2,)` in `[-1, 1]²`
A continuous velocity command; the env scales it by `max_speed` (default 12 m/s)
and integrates for `dt` (default 0.5 s).

### Reward
| Term | Value |
|---|---|
| Progress | `+ (previous distance − new distance)` metres closed toward the goal |
| Time cost | `− 0.02` per step |
| Collision | `− 0.5` (building, campus edge, or disruption zone) |
| Success | `+ 25` and **terminate** |

The dense progress term is the shaping signal; the sparse success bonus is the
true objective. Collision is handled by *sliding* (try full move → x-only →
y-only → stay), so the agent brushes along walls instead of sticking.

### Episode
* Terminates: goal reached (within `success_radius`, default 8 m).
* Truncates: `max_steps` (default 300) reached.

### Task sets (deterministic, like PortSimEnv)
* **1,000 training tasks**, seed `20261015`.
* **50 evaluation tasks**, seed `20261016`.
* Each task = a (start POI, goal POI) pair at least 120 m apart.
* Generation: `campus_sim/tasks.py::generate_tasks`.

### Constraints & disruptions (optional, off by default)
`n_disruptions=k` places *k* circular closed zones on the route each episode,
seeded — the analogue of PortSimEnv's "cranes go offline / weather" disruptions.
`gust>0` adds stochastic velocity perturbation for domain randomisation.

### Success metric
**Success rate** = fraction of evaluation tasks where the goal is reached, plus
**mean episode return**. Both are reported by the training scripts.

### Reference numbers (50 eval tasks)
| Agent | Success rate |
|---|---|
| Random | 0% |
| CEM (linear policy, numpy) | ~12% |
| PPO (150k steps) | ~36% |
| Expert steering heuristic | 100% |

---

## Part 3 — Template to copy for your own environment

```
ENVIRONMENT SPEC — <name>

Identity
  id:            <name>-v0
  domain:        <e.g. continuous navigation / discrete scheduling>
  api:           Gymnasium (reset/step)
  determinism:   seeded; (seed, task_id) -> identical episode

Real-world grounding
  source of world:   <OSM / operator data / CSV / survey>
  what is real:      <list>
  what is modelled:  <list + why>

Task
  <one paragraph: who is the agent, what must it achieve, what makes it hard>

Observation   <shape, fields, ranges, scaling>
Action        <shape, ranges, semantics>
Reward        <every term and its value; dense vs sparse>
Episode       <termination + truncation conditions>
Task sets     <n_train, n_eval, seeds, generation rule>
Constraints / disruptions   <what, how introduced, seeded?>
Success metric  <the number you report>

Verifiability
  <why the reward cannot be gamed; what "correct" means>

Viewer (presentation only, never affects the sim)
  <what it renders; where the recorded state comes from>
```

---

## Part 4 — Adapting this to the port-style problem

The port environment is not a physics sim — it is an **operations** sim. The
same five steps produce it:

* **Operation:** berth planning. **Outcome:** total vessel waiting / turnaround time.
* **World data:** berth layout + vessel arrival table (`data/*.json`).
* **Observation:** vessel queue, berth occupancy, time.
* **Action:** assign the next vessel to a berth (discrete), or a schedule (multi-step).
* **Reward:** `− waiting time − idle berth cost − violation penalty`.
* **Determinism:** same arrivals seed → same scenario.
* **Viewer:** draw vessels moving to berths, replay the schedule.

The 3D viewer in this repo (`campus_sim/export.py`) is generic enough to reuse:
it renders polygons, paths, moving points and markers from a JSON rollout — swap
the shapes and you have a port view.
