"""CampusSim -- an RL environment that rebuilds the IKGPTU Kapurthala campus as a
deterministic simulation, with a 3D viewer for every rollout."""
from .geometry import Campus, load_campus, point_in_polygon  # noqa: F401
from .tasks import build_task_sets, generate_tasks            # noqa: F401
from .campus_env import CampusNavEnv                          # noqa: F401

__all__ = ["Campus", "load_campus", "point_in_polygon",
           "build_task_sets", "generate_tasks", "CampusNavEnv"]
__version__ = "0.1.0"
