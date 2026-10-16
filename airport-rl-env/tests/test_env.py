"""Tests for the Airport RL environment.

Run with::

    pip install pytest
    pytest -q
"""

import math

import numpy as np
import gymnasium as gym
import mujoco

import airport_rl  # noqa: F401  (registers the env)

TASKS = ("navigate", "dock_aircraft", "shuttle", "patrol")


def make_env(task="navigate"):
    return gym.make("AirportGround-v0", task=task).unwrapped


def scripted_action(raw):
    """Same simple goal-seeking + lidar-avoidance driver used by the demo."""
    goal = raw._goal
    dx = float(goal[0]) - raw._x()
    dy = float(goal[1]) - raw._y()
    dist = math.hypot(dx, dy)
    yaw = raw._yaw()
    bearing = (math.atan2(dy, dx) - yaw + math.pi) % (2.0 * math.pi) - math.pi
    ax, ay = math.cos(bearing), math.sin(bearing)
    lidar = raw._lidar()
    n = len(lidar)
    rx = ry = 0.0
    for i, r in enumerate(lidar):
        if r < 0.28:
            a = -math.pi + i * (2.0 * math.pi / n)
            w = (0.28 - r) / 0.28
            rx -= w * math.cos(a)
            ry -= w * math.sin(a)
    desired = math.atan2(ay + 1.4 * ry, ax + 1.4 * rx)
    steer = float(np.clip(desired * 1.7, -1.0, 1.0))
    throttle = 1.0 if abs(desired) < 0.9 else 0.5
    if dist < raw.cfg.goal_radius * 1.5:
        throttle = min(throttle, 0.6)
    return np.array([throttle, steer], dtype=np.float32)


# ----------------------------------------------------------------------
def test_reset_shapes_all_tasks():
    for task in TASKS:
        env = make_env(task)
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs), f"{task}: obs outside space"
        assert info["task"] == task
        assert info["n_waypoints"] >= 1
        assert env.action_space.shape == (2,)
        env.close()


def test_step_contract():
    env = make_env()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert {"is_success", "collision", "distance", "waypoint", "task"}.issubset(info)
    env.close()


def test_seed_determinism():
    a, b = make_env(), make_env()
    o1, i1 = a.reset(seed=42)
    o2, i2 = b.reset(seed=42)
    assert np.allclose(o1, o2)
    assert i1["gate_index"] == i2["gate_index"]
    for _ in range(20):
        act = np.array([0.3, 0.1], dtype=np.float32)
        assert abs(a.step(act)[1] - b.step(act)[1]) < 1e-9
    a.close()
    b.close()


def test_collision_terminates():
    env = make_env()
    env.reset(seed=2)
    env._set_free_body_pose(env.agent_body, (0.0, 0.0, 0.5), np.pi / 2)
    mujoco.mj_forward(env.model, env.data)
    collided = False
    for _ in range(120):
        _, _, terminated, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        if info["collision"]:
            collided = True
            assert terminated
            break
    assert collided, "expected a collision when driving into the terminal"
    env.close()


def test_every_mission_is_solvable():
    """A simple scripted driver must be able to complete each mission."""
    for task in TASKS:
        env = make_env(task)
        successes = 0
        for seed in range(3):
            env.reset(seed=seed)
            for _ in range(env.cfg.max_steps):
                _, _, terminated, truncated, info = env.step(scripted_action(env))
                if terminated or truncated:
                    break
            successes += int(info["is_success"])
        assert successes >= 1, f"task {task!r} was never completed"
        env.close()


def test_taxi_aircraft_move():
    env = make_env()
    env.reset(seed=0)
    p0 = env.data.qpos[env.model.jnt_qposadr[env.model.body_jntadr[env.taxi_bodies[0]]]:][:2].copy()
    for _ in range(60):
        env.step(np.zeros(2, dtype=np.float32))
    p1 = env.data.qpos[env.model.jnt_qposadr[env.model.body_jntadr[env.taxi_bodies[0]]]:][:2]
    assert not np.allclose(p0, p1), "taxiing aircraft did not move"
    env.close()


def test_lidar_in_range():
    env = make_env()
    env.reset(seed=0)
    lidar = env._lidar()
    assert lidar.shape == (env.cfg.n_rays,)
    assert lidar.min() >= 0.0 and lidar.max() <= 1.0
    env.close()
