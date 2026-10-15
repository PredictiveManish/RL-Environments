"""
campus_env.py
=============
IKGPTU-CampusNav -- a Gymnasium reinforcement-learning environment in which an
autonomous campus shuttle must drive between buildings of the I. K. Gujral
Punjab Technical University main campus, Kapurthala, without hitting anything.

The design deliberately follows the "real-world work turned into a deterministic
simulation" pattern of the PortSimEnv reference: a seeded task set, a physical
state the agent acts on, shaped rewards tied to the operational outcome, and an
optional disruption layer.

Task
----
Given a start location and a goal building, drive the shuttle to the goal.

Observation (Box, 8,)  all values roughly in [-1, 1]
    [0] agent x / S          [4] vx / vmax
    [1] agent y / S          [5] vy / vmax
    [2] goal  x / S          [6] sin(bearing to goal)
    [3] goal  y / S          [7] cos(bearing to goal)
    S = campus half-diagonal (metres), vmax = max speed (m/s)

Action (Box, 2,)  continuous velocity command in [-1, 1]^2 -> scaled by vmax.

Reward
    + progress toward goal (metres closed this step)
    - 0.02 per step (time pressure)
    - 0.5 on collision with a building / campus edge / disruption zone
    + 25 and terminate on reaching the goal (within success_radius)

Episode ends on success, or truncates at max_steps.
"""
from __future__ import annotations

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .geometry import Campus, load_campus
from .tasks import build_task_sets


class CampusNavEnv(gym.Env):
    metadata = {"render_modes": ["state"], "render_fps": 30}

    def __init__(self, campus: Campus | None = None,
                 task_set: str = "train", tasks=None,
                 max_steps: int = 300, dt: float = 0.5, max_speed: float = 12.0,
                 agent_radius: float = 2.0, success_radius: float = 8.0,
                 n_disruptions: int = 0, disruption_radius: float = 25.0,
                 gust: float = 0.0):
        super().__init__()
        self.campus = campus or load_campus()
        self.dt = dt
        self.max_speed = max_speed
        self.agent_radius = agent_radius
        self.success_radius = success_radius
        self.max_steps = max_steps
        self.n_disruptions = n_disruptions
        self.disruption_radius = disruption_radius
        self.gust = gust  # std-dev of random acceleration (m/s^2); 0 = deterministic

        if tasks is None:
            tasks = build_task_sets(self.campus)[task_set]
        self.tasks = tasks

        S = self.campus.scale
        self._S = S
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # episode state
        self.pos = np.zeros(2, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.goal = np.zeros(2, dtype=float)
        self.task = None
        self.dynamic_obstacles = []  # list of (cx, cy, r)
        self.steps = 0
        self.cum_reward = 0.0
        self.n_collisions = 0
        self.trajectory = []

    # ------------------------------------------------------------------ #
    def _blocked(self, x, y) -> bool:
        if self.campus.collides(x, y, self.agent_radius):
            return True
        for (cx, cy, r) in self.dynamic_obstacles:
            if math.hypot(x - cx, y - cy) < r + self.agent_radius:
                return True
        return False

    def _obs(self) -> np.ndarray:
        S = self._S
        dx, dy = self.goal - self.pos
        d = math.hypot(dx, dy) or 1e-9
        return np.array([
            self.pos[0] / S, self.pos[1] / S,
            self.goal[0] / S, self.goal[1] / S,
            self.vel[0] / self.max_speed, self.vel[1] / self.max_speed,
            dx / d, dy / d,
        ], dtype=np.float32)

    def _make_disruptions(self, rng):
        """Place circular closed zones roughly on the route, avoiding endpoints."""
        obs = []
        if self.n_disruptions <= 0:
            return obs
        mid = (self.pos + self.goal) / 2.0
        for _ in range(self.n_disruptions):
            for _try in range(50):
                ang = rng.uniform(0, 2 * math.pi)
                rad = rng.uniform(0, 0.5) * np.linalg.norm(self.goal - self.pos)
                cx = mid[0] + rad * math.cos(ang)
                cy = mid[1] + rad * math.sin(ang)
                if (math.hypot(cx - self.pos[0], cy - self.pos[1]) < 60 or
                        math.hypot(cx - self.goal[0], cy - self.goal[1]) < 60):
                    continue
                if not self.campus.inside_boundary(cx, cy):
                    continue
                if self.campus.collides(cx, cy, self.disruption_radius):
                    continue
                obs.append((float(cx), float(cy), float(self.disruption_radius)))
                break
        return obs

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "task_id" in options:
            self.task = self.tasks[options["task_id"] % len(self.tasks)]
        else:
            self.task = self.tasks[int(self.np_random.integers(len(self.tasks)))]

        self.pos = np.array(self.task["start"], dtype=float)
        self.goal = np.array(self.task["goal"], dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.steps = 0
        self.cum_reward = 0.0
        self.n_collisions = 0
        self.trajectory = [self.pos.copy()]
        self.dynamic_obstacles = self._make_disruptions(self.np_random)
        info = {"task_id": self.task["id"],
                "start_name": self.task["start_name"],
                "goal_name": self.task["goal_name"]}
        return self._obs(), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.vel = action * self.max_speed
        if self.gust > 0:
            self.vel = self.vel + self.np_random.normal(0, self.gust, size=2)

        px, py = self.pos
        dx, dy = self.vel * self.dt
        collided = False
        if not self._blocked(px + dx, py + dy):
            nx, ny = px + dx, py + dy
        elif not self._blocked(px + dx, py):          # slide along x
            nx, ny, collided = px + dx, py, True
        elif not self._blocked(px, py + dy):          # slide along y
            nx, ny, collided = px, py + dy, True
        else:
            nx, ny, collided = px, py, True

        prev_d = math.hypot(*(self.goal - self.pos))
        self.pos = np.array([nx, ny], dtype=float)
        dist = math.hypot(*(self.goal - self.pos))
        progress = prev_d - dist

        reward = progress - 0.02
        if collided:
            reward -= 0.5
            self.n_collisions += 1

        self.steps += 1
        terminated = False
        if dist < self.success_radius:
            reward += 25.0
            terminated = True
        truncated = self.steps >= self.max_steps

        self.cum_reward += reward
        self.trajectory.append(self.pos.copy())
        info = {"distance": float(dist), "collision": bool(collided),
                "success": bool(terminated), "task_id": self.task["id"],
                "position": self.pos.tolist()}
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    def get_state(self):
        """Full physical state -- consumed by the 3D viewer / rollout exporter."""
        return {
            "task_id": self.task["id"] if self.task else None,
            "goal": self.goal.tolist(),
            "start": self.trajectory[0].tolist() if self.trajectory else None,
            "goal_name": self.task["goal_name"] if self.task else None,
            "start_name": self.task["start_name"] if self.task else None,
            "position": self.pos.tolist(),
            "trajectory": [p.tolist() for p in self.trajectory],
            "dynamic_obstacles": [list(o) for o in self.dynamic_obstacles],
            "steps": self.steps,
            "cum_reward": float(self.cum_reward),
            "n_collisions": int(self.n_collisions),
        }

    def render(self):
        return self.get_state()


gym.register(id="IKGPTU-CampusNav-v0", entry_point="campus_sim.campus_env:CampusNavEnv")
