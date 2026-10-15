"""Airport RL - a MuJoCo + Gymnasium environment for airport ground operations.

Quick start::

    import gymnasium as gym
    import airport_rl  # registers "AirportGround-v0"

    env = gym.make("AirportGround-v0", render_mode="human")
    obs, info = env.reset()
    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
"""

from .envs.airport_env import AirportGroundEnv
from .envs.config import AirportConfig, EnvConfig

__version__ = "1.0.0"
__all__ = ["AirportGroundEnv", "AirportConfig", "EnvConfig"]


def _register():
    from gymnasium.envs.registration import register

    if "AirportGround-v0" not in _registered_ids():
        register(
            id="AirportGround-v0",
            entry_point="airport_rl.envs.airport_env:AirportGroundEnv",
            max_episode_steps=None,  # the env manages truncation itself
        )


def _registered_ids():
    try:
        from gymnasium.envs.registration import registry

        return set(registry.keys())
    except Exception:
        return set()


_register()
