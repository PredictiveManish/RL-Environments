"""Basic tests for the Airport RL environment.

Run with::

    pip install pytest
    pytest -q
"""

import numpy as np
import gymnasium as gym

import airport_rl  # noqa: F401  (registers the env)


def make_env():
    return gym.make("AirportGround-v0").unwrapped


def test_reset_shapes():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), "obs outside observation space"
    assert env.action_space.shape == (2,)
    assert "distance" in info
    env.close()


def test_step_contract():
    env = make_env()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert set(["is_success", "collision", "distance"]).issubset(info)
    env.close()


def test_seed_determinism():
    a = make_env()
    b = make_env()
    o1, i1 = a.reset(seed=42)
    o2, i2 = b.reset(seed=42)
    assert np.allclose(o1, o2)
    assert i1["gate_index"] == i2["gate_index"]
    for _ in range(20):
        act = np.array([0.3, 0.1], dtype=np.float32)
        r1 = a.step(act)[1]
        r2 = b.step(act)[1]
        assert abs(r1 - r2) < 1e-9
    a.close()
    b.close()


def test_collision_terminates():
    env = make_env()
    env.reset(seed=2)
    # Aim the vehicle at the terminal wall and floor it.
    env._set_free_body_pose(env.agent_body, (0.0, 0.0, 0.5), np.pi / 2)
    import mujoco

    mujoco.mj_forward(env.model, env.data)
    collided = False
    for _ in range(80):
        _, _, terminated, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        if info["collision"]:
            collided = True
            assert terminated
            break
    assert collided, "expected a collision when driving into the terminal"
    env.close()


def test_success_terminates():
    env = make_env()
    env.reset(seed=3)
    goal = env._goal.copy()
    env._set_free_body_pose(env.agent_body, (goal[0] - 10.0, goal[1], 0.5), 0.0)
    import mujoco

    mujoco.mj_forward(env.model, env.data)
    env._prev_dist = env._distance_to_goal()
    reached = False
    for _ in range(120):
        dx, dy = goal[0] - env._x(), goal[1] - env._y()
        yaw = env._yaw()
        bearing = (np.arctan2(dy, dx) - yaw + np.pi) % (2 * np.pi) - np.pi
        _, _, terminated, _, info = env.step(
            np.array([1.0, np.clip(bearing, -1, 1)], dtype=np.float32)
        )
        if info["is_success"]:
            reached = True
            assert terminated
            break
    assert reached, "expected the goal to be reachable"
    env.close()


def test_lidar_in_range():
    env = make_env()
    env.reset(seed=0)
    lidar = env._lidar()
    assert lidar.shape == (env.cfg.n_rays,)
    assert lidar.min() >= 0.0 and lidar.max() <= 1.0
    env.close()
