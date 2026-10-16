"""The Airport RL environment (Gymnasium + MuJoCo).

Missions
--------
A ground vehicle (a "baggage tug" / pushback tractor) works the airport apron.
Pick the job with ``task=`` (see ``EnvConfig.task``):

* ``"navigate"``       - drive to a gate stand (point-to-point).
* ``"dock_aircraft"``  - dock precisely at the nose of a parked aircraft.
* ``"shuttle"``        - two-leg delivery: baggage hall -> gate stand.
* ``"patrol"``         - follow a multi-waypoint route along the apron lanes.

The scene also contains live elements the vehicle must work around: parked
aircraft and turnaround trucks at every gate, ambient ground traffic, and
aircraft taxiing along the parallel taxiway.

Observation
-----------
Flat vector: relative goal (current waypoint) info + ego state + lidar ring.
Optionally an RGB image (see ``EnvConfig.include_rgb``).

Action
------
``Box(2,)`` = [throttle, steering], each in [-1, 1].

Reward
------
Progress toward the current waypoint, a heading bonus, a small time penalty,
a bonus per intermediate waypoint, a big bonus on completing the mission, and a
big penalty (with termination) on collision.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

try:  # MuJoCo is imported lazily-ish so the module can be inspected without a GPU
    import mujoco
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This environment needs MuJoCo. Install it with:  pip install mujoco"
    ) from exc

from ..scene.airport_builder import AirportScene
from ..utils.math_utils import (
    lidar_directions,
    quat_from_yaw,
    relative_polar,
    wrap_angle,
    yaw_from_quat,
)
from .config import AirportConfig, EnvConfig

VALID_TASKS = ("navigate", "dock_aircraft", "shuttle", "patrol")


class AirportGroundEnv(gym.Env):
    """Gymnasium environment: airport ground-vehicle operations."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        airport: Optional[AirportConfig] = None,
        config: Optional[EnvConfig] = None,
        task: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.airport = airport or AirportConfig()
        self.render_mode = render_mode
        self.task = task or self.cfg.task
        if self.task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got {self.task!r}")

        # ---- Build the scene ------------------------------------------
        scene = AirportScene(self.airport)
        xml, meta = scene.build()
        self._xml = xml
        self._meta = meta
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # ---- Resolve ids ----------------------------------------------
        self.agent_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent")
        self.agent_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "agent_joint")
        self.agent_qpos_adr = self.model.jnt_qposadr[self.agent_joint]
        self.agent_qvel_adr = self.model.jnt_dofadr[self.agent_joint]

        self.mover_bodies = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"mover_{i}")
            for i in range(self.airport.n_movers)
        ]
        self.taxi_bodies = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"taxi_{i}")
            for i in range(self.airport.n_taxi_aircraft)
        ]

        # Geoms whose contacts with the agent are "drivable" (ignored)
        self.driveable_geoms = set()
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith(self._meta["driveable_prefix"]):
                self.driveable_geoms.add(gid)

        self.ray_dirs = lidar_directions(self.cfg.n_rays)

        # ---- Spaces ----------------------------------------------------
        obs_dim = 6 + (self.cfg.n_rays if self.cfg.include_lidar else 0)
        if self.cfg.include_rgb:
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(-1.0, 1.0, (6,), np.float32),
                    "lidar": spaces.Box(0.0, 1.0, (self.cfg.n_rays,), np.float32),
                    "image": spaces.Box(
                        0, 255, (self.cfg.rgb_height, self.cfg.rgb_width, 3), np.uint8
                    ),
                }
            )
        else:
            self.observation_space = spaces.Box(-1.0, 1.0, (obs_dim,), np.float32)

        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

        # ---- Episode state --------------------------------------------
        self._step_count = 0
        self._goal = np.zeros(2, np.float32)
        self._route: List[Tuple[float, float]] = []
        self._wp = 0
        self._wp_radius = self.cfg.goal_radius
        self._prev_dist = 0.0
        self._t = 0.0
        self._active_task = self.task

        # ---- Rendering -------------------------------------------------
        self._viewer = None
        self._renderer = None
        self._camera = None

    # ==================================================================
    # Gymnasium API
    # ==================================================================
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self._step_count = 0
        self._t = 0.0

        mujoco.mj_resetData(self.model, self.data)

        # Start pose (small randomisation around the configured start)
        sx, sy, sz = self.airport.agent_start
        if self.cfg.spawn_radius > 0:
            sx += self.np_random.uniform(-self.cfg.spawn_radius, self.cfg.spawn_radius)
            sy += self.np_random.uniform(-self.cfg.spawn_radius, self.cfg.spawn_radius)
        yaw0 = float(self.np_random.uniform(-np.pi, np.pi))
        self._set_free_body_pose(self.agent_body, (sx, sy, sz), yaw0)

        # Which gate this episode is about (forced via ``options`` if given)
        gates = self._meta["gate_positions"]
        if options and "gate_index" in options:
            gi = int(options["gate_index"]) % len(gates)
        else:
            gi = int(self.np_random.integers(len(gates)))
        self._gate_index = gi

        # Mission can be overridden per-episode
        self._active_task = (options or {}).get("task", self.task)
        if self._active_task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}")

        self._route = self._build_route(self._active_task, gi)
        self._wp_radius = self._task_radius(self._active_task)
        self._wp = 0
        self._goal = np.array(self._route[0], dtype=np.float32)

        self._update_moving(0.0)
        mujoco.mj_forward(self.model, self.data)

        self._prev_dist = self._distance_to_goal()
        obs = self._get_obs()
        info = {
            "task": self._active_task,
            "gate_index": gi,
            "n_waypoints": len(self._route),
            "distance": float(self._prev_dist),
        }
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)
        throttle, steer = float(action[0]), float(action[1])

        speed = throttle * (self.cfg.max_speed if throttle >= 0 else self.cfg.max_reverse_speed)
        yaw_rate = steer * self.cfg.max_yaw_rate

        for _ in range(self.cfg.frame_skip):
            self._drive_agent(speed, yaw_rate)
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        self._t += self.cfg.frame_skip * self.cfg.sim_dt
        self._update_moving(self._t)

        dist = self._distance_to_goal()
        collision = self._check_collision()

        # ---- Reward ----------------------------------------------------
        reward = self.cfg.w_progress * (self._prev_dist - dist)
        reward += self.cfg.w_step
        _, bearing = relative_polar(self._goal[0] - self._x(), self._goal[1] - self._y(), self._yaw())
        reward += self.cfg.w_heading * np.cos(bearing) * (1.0 if speed >= 0 else -1.0)

        terminated = False
        truncated = False
        is_success = False

        if collision:
            reward += self.cfg.w_collision
            terminated = True
        elif dist < self._wp_radius:
            # Reached this waypoint - advance the mission
            self._wp += 1
            if self._wp >= len(self._route):
                reward += self.cfg.w_success
                is_success = True
                terminated = True
            else:
                reward += self.cfg.w_waypoint
                self._goal = np.array(self._route[self._wp], dtype=np.float32)
                dist = self._distance_to_goal()
        elif self._step_count >= self.cfg.max_steps:
            truncated = True

        self._prev_dist = dist
        obs = self._get_obs()
        info = {
            "task": self._active_task,
            "is_success": is_success,
            "collision": collision,
            "distance": float(dist),
            "waypoint": self._wp,
            "n_waypoints": len(self._route),
            "gate_index": self._gate_index,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        if self.render_mode == "human":
            return self._render_human()
        return None

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ==================================================================
    # Missions
    # ==================================================================
    def _build_route(self, task: str, gi: int) -> List[Tuple[float, float]]:
        if task == "navigate":
            return [tuple(self._meta["gate_positions"][gi])]
        if task == "dock_aircraft":
            ax, ay = self._meta["aircraft_positions"][gi]
            return [(float(ax), float(ay) - self.airport.dock_offset)]
        if task == "shuttle":
            return [tuple(self._meta["baggage_dock"]), tuple(self._meta["gate_positions"][gi])]
        if task == "patrol":
            return [tuple(p) for p in self._meta["patrol_route"]]
        raise ValueError(f"unknown task {task!r}")

    def _task_radius(self, task: str) -> float:
        if task == "dock_aircraft":
            return self.cfg.dock_radius
        if task == "patrol":
            return self.cfg.patrol_radius
        return self.cfg.goal_radius

    # ==================================================================
    # Internals
    # ==================================================================
    def _drive_agent(self, speed: float, yaw_rate: float):
        """Set the agent free-joint velocity (world-frame unicycle)."""
        yaw = self._yaw()
        adr = self.agent_qvel_adr
        self.data.qvel[adr + 0] = speed * np.cos(yaw)
        self.data.qvel[adr + 1] = speed * np.sin(yaw)
        self.data.qvel[adr + 2] = 0.0
        self.data.qvel[adr + 3] = 0.0
        self.data.qvel[adr + 4] = 0.0
        self.data.qvel[adr + 5] = yaw_rate
        self.data.qpos[self.agent_qpos_adr + 2] = self.airport.agent_start[2]
        self.data.qpos[self.agent_qpos_adr + 3:self.agent_qpos_adr + 7] = quat_from_yaw(yaw)

    def _set_free_body_pose(self, body_id: int, pos, yaw: float):
        jid = self.model.body_jntadr[body_id]
        adr = self.model.jnt_qposadr[jid]
        self.data.qpos[adr + 0] = pos[0]
        self.data.qpos[adr + 1] = pos[1]
        self.data.qpos[adr + 2] = pos[2]
        self.data.qpos[adr + 3:adr + 7] = quat_from_yaw(yaw)

    def _pingpong(self, p0, p1, speed, t):
        """Position + heading along a ping-pong segment at time ``t``."""
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        seg = p1 - p0
        length = float(np.hypot(*seg)) + 1e-6
        period = 2.0 * length / max(speed, 1e-6)
        phase = (t % period) / period
        u = phase if phase <= 0.5 else 1.0 - phase
        pos = p0 + u * seg
        direction = seg if phase <= 0.5 else -seg
        yaw = float(np.arctan2(direction[1], direction[0]))
        return pos, yaw

    def _zero_qvel(self, body_id: int):
        jid = self.model.body_jntadr[body_id]
        adr = self.model.jnt_dofadr[jid]
        self.data.qvel[adr:adr + 6] = 0.0

    def _update_moving(self, t: float):
        """Animate ambient traffic and taxiing aircraft."""
        # Ground traffic
        for i, (p0, p1) in enumerate(self._meta["mover_waypoints"]):
            pos, yaw = self._pingpong(p0, p1, self.airport.mover_speed, t)
            self._set_free_body_pose(
                self.mover_bodies[i], (pos[0], pos[1], self.airport.mover_half[2]), yaw
            )
            self._zero_qvel(self.mover_bodies[i])
        # Taxiing aircraft on the parallel taxiway
        for i, (p0, p1) in enumerate(self._meta["taxi_routes"]):
            pos, yaw = self._pingpong(p0, p1, self.airport.taxi_aircraft_speed, t)
            self._set_free_body_pose(
                self.taxi_bodies[i], (pos[0], pos[1], self.airport.aircraft_z), yaw
            )
            self._zero_qvel(self.taxi_bodies[i])
        mujoco.mj_forward(self.model, self.data)

    def _x(self) -> float:
        return float(self.data.qpos[self.agent_qpos_adr + 0])

    def _y(self) -> float:
        return float(self.data.qpos[self.agent_qpos_adr + 1])

    def _yaw(self) -> float:
        q = self.data.qpos[self.agent_qpos_adr + 3:self.agent_qpos_adr + 7]
        return yaw_from_quat(q)

    def _speed(self) -> float:
        adr = self.agent_qvel_adr
        return float(np.hypot(self.data.qvel[adr], self.data.qvel[adr + 1]))

    def _distance_to_goal(self) -> float:
        return float(np.hypot(self._goal[0] - self._x(), self._goal[1] - self._y()))

    def _check_collision(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if b1 == self.agent_body or b2 == self.agent_body:
                other = c.geom2 if b1 == self.agent_body else c.geom1
                if other in self.driveable_geoms:
                    continue
                if b1 == b2:  # self-contact
                    continue
                return True
        return False

    def _lidar(self) -> np.ndarray:
        cfg = self.cfg
        origin = np.array([self._x(), self._y(), cfg.lidar_height], dtype=np.float64)
        yaw = self._yaw()
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        geomid = np.zeros(1, dtype=np.int32)
        out = np.ones(cfg.n_rays, dtype=np.float32)
        for i in range(cfg.n_rays):
            d2 = rot @ self.ray_dirs[i]
            vec = np.array([d2[0], d2[1], 0.0], dtype=np.float64)
            dist = mujoco.mj_ray(
                self.model, self.data, origin, vec, None, 1, self.agent_body, geomid
            )
            if dist < 0 or dist > cfg.lidar_max_range:
                dist = cfg.lidar_max_range
            out[i] = np.clip(dist / cfg.lidar_max_range, 0.0, 1.0)
        return out

    def _get_obs(self):
        dx = self._goal[0] - self._x()
        dy = self._goal[1] - self._y()
        dist, bearing = relative_polar(dx, dy, self._yaw())
        speed = self._speed()
        yaw = self._yaw()

        state = np.array(
            [
                np.clip(speed / self.cfg.max_speed, -1.0, 1.0),
                np.sin(yaw),
                np.cos(yaw),
                np.clip(dist / self.cfg.lidar_max_range, 0.0, 1.0) * 2.0 - 1.0,
                np.sin(bearing),
                np.cos(bearing),
            ],
            dtype=np.float32,
        )
        lidar = self._lidar() if self.cfg.include_lidar else np.zeros(0, np.float32)

        if self.cfg.include_rgb:
            return {"state": state, "lidar": lidar, "image": self._render_rgb()}
        return np.concatenate([state, lidar]).astype(np.float32)

    # ------------------------------------------------------------------
    def _make_camera(self):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = self.agent_body
        cam.distance = self.cfg.cam_distance
        cam.elevation = self.cfg.cam_elevation
        cam.azimuth = self.cfg.cam_azimuth
        return cam

    def _render_rgb(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, self.cfg.rgb_height, self.cfg.rgb_width
            )
            self._camera = self._make_camera()
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render().copy()

    def _render_human(self):
        import mujoco.viewer

        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._camera = self._make_camera()
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._viewer.cam.trackbodyid = self.agent_body
            self._viewer.cam.distance = self.cfg.cam_distance
            self._viewer.cam.elevation = self.cfg.cam_elevation
            self._viewer.cam.azimuth = self.cfg.cam_azimuth
        self._viewer.sync()
