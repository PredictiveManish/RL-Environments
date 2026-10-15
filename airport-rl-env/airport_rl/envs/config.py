"""Configuration for the Airport RL environment.

All distances are in metres, angles in radians, and the world uses MuJoCo's
default Z-up convention (X/Y = ground plane, Z = up).

The layout below is a stylised, *compressed* model of a single-runway
international airport (the proportions loosely follow Noida International
"Jewar" Airport: one long runway, a parallel taxiway, an apron in front of a
single terminal, plus a control tower and airside service facilities). It is
deliberately compressed so the whole scene fits comfortably in a simulation and
renders well, while keeping recognisable real-world structure.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AirportConfig:
    """Geometry of the airport scene (all values in metres)."""

    name: str = "Noida International Airport (Jewar) - simulated"

    # ---- Movement area -------------------------------------------------
    # Runway: long, thin strip. half-extents = (length/2, width/2, thickness/2)
    runway_pos: Tuple[float, float, float] = (0.0, -120.0, 0.05)
    runway_half: Tuple[float, float, float] = (150.0, 11.0, 0.05)

    # Parallel taxiway connecting runway to apron
    taxiway_pos: Tuple[float, float, float] = (0.0, -75.0, 0.05)
    taxiway_half: Tuple[float, float, float] = (150.0, 8.0, 0.05)

    # Apron / tarmac: the region where ground vehicles actually work
    apron_pos: Tuple[float, float, float] = (0.0, -28.0, 0.05)
    apron_half: Tuple[float, float, float] = (150.0, 30.0, 0.05)

    # ---- Buildings -----------------------------------------------------
    terminal_pos: Tuple[float, float, float] = (0.0, 48.0, 6.0)
    terminal_half: Tuple[float, float, float] = (92.0, 16.0, 6.0)

    tower_pos: Tuple[float, float, float] = (-110.0, 46.0, 17.0)
    tower_radius: float = 4.0
    tower_height: float = 34.0

    # ---- Gates ---------------------------------------------------------
    # X positions of gate stands along the apron edge, and the Y of the
    # parking line. Each gate holds one parked aircraft, nose facing the
    # apron (-Y). The "stand" the agent must reach sits just ahead of the nose.
    gate_xs: Tuple[float, ...] = (-100.0, -60.0, -20.0, 20.0, 60.0, 100.0)
    gate_y: float = -8.0
    aircraft_z: float = 2.6  # ride height so engines clear the ground

    # ---- Airside service facilities ------------------------------------
    baggage_pos: Tuple[float, float, float] = (126.0, -28.0, 3.0)
    baggage_half: Tuple[float, float, float] = (12.0, 9.0, 3.0)

    fuel_pos: Tuple[float, float, float] = (-126.0, -28.0, 2.0)
    fuel_radius: float = 5.0
    fuel_height: float = 4.0

    maintenance_pos: Tuple[float, float, float] = (0.0, -55.0, 2.5)
    maintenance_half: Tuple[float, float, float] = (14.0, 5.0, 2.5)

    # ---- Agent (the ground vehicle the RL policy drives) ---------------
    # A "baggage tug" / pushback tractor. Half-extents of its chassis box.
    agent_half: Tuple[float, float, float] = (1.6, 0.9, 0.5)
    agent_start: Tuple[float, float, float] = (-110.0, -42.0, 0.5)

    # ---- Moving obstacles (other ground traffic) -----------------------
    n_movers: int = 4
    mover_half: Tuple[float, float, float] = (1.8, 1.0, 0.5)
    mover_speed: float = 5.0  # m/s patrol speed

    # Colours (rgba 0-1)
    col_ground: Tuple[float, float, float, float] = (0.20, 0.22, 0.24, 1.0)
    col_runway: Tuple[float, float, float, float] = (0.10, 0.10, 0.12, 1.0)
    col_taxiway: Tuple[float, float, float, float] = (0.16, 0.16, 0.18, 1.0)
    col_apron: Tuple[float, float, float, float] = (0.30, 0.31, 0.33, 1.0)
    col_terminal: Tuple[float, float, float, float] = (0.72, 0.76, 0.80, 1.0)
    col_tower: Tuple[float, float, float, float] = (0.85, 0.86, 0.88, 1.0)
    col_marking: Tuple[float, float, float, float] = (0.95, 0.95, 0.95, 1.0)
    col_aircraft: Tuple[float, float, float, float] = (0.90, 0.92, 0.95, 1.0)
    col_agent: Tuple[float, float, float, float] = (0.10, 0.70, 0.95, 1.0)
    col_mover: Tuple[float, float, float, float] = (0.95, 0.45, 0.10, 1.0)
    col_baggage: Tuple[float, float, float, float] = (0.55, 0.42, 0.25, 1.0)
    col_fuel: Tuple[float, float, float, float] = (0.75, 0.20, 0.20, 1.0)


@dataclass
class EnvConfig:
    """Reinforcement-learning / simulation parameters."""

    # Simulation
    sim_dt: float = 0.01
    frame_skip: int = 5  # -> control step of 0.05 s
    max_steps: int = 600

    # Agent kinematics (unicycle model)
    max_speed: float = 9.0       # m/s forward
    max_reverse_speed: float = 3.0
    max_yaw_rate: float = 1.6    # rad/s

    # Task
    goal_radius: float = 4.0     # distance at which the agent "arrives"
    spawn_radius: float = 6.0    # randomisation of start/goal
    collision_radius: float = 1.2  # extra padding for collision checks

    # Lidar
    n_rays: int = 24
    lidar_max_range: float = 45.0
    lidar_height: float = 0.6

    # Reward weights
    w_progress: float = 1.0
    w_success: float = 50.0
    w_collision: float = -50.0
    w_step: float = -0.02
    w_heading: float = 0.15

    # Observation
    include_lidar: bool = True
    include_rgb: bool = False
    rgb_width: int = 160
    rgb_height: int = 90

    # Camera for the viewer / rgb render
    cam_distance: float = 26.0
    cam_elevation: float = -55.0
    cam_azimuth: float = -90.0
