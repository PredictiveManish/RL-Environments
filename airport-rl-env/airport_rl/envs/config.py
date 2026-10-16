"""Configuration for the Airport RL environment.

All distances are in metres, angles in radians, and the world uses MuJoCo's
default Z-up convention (X/Y = ground plane, Z = up).

The layout is a stylised, *compressed* model of a single-runway international
airport (proportions loosely following Noida International "Jewar" Airport): a
runway, a parallel taxiway with live taxiing aircraft, an apron in front of a
terminal with six gate stands (each with a parked aircraft and turnaround
service vehicles), a control tower, and airside facilities (baggage hall, fuel
farm, maintenance hangar).
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class AirportConfig:
    """Geometry of the airport scene (all values in metres)."""

    name: str = "Noida International Airport (Jewar) - simulated"

    # ---- Movement area -------------------------------------------------
    runway_pos: Tuple[float, float, float] = (0.0, -120.0, 0.05)
    runway_half: Tuple[float, float, float] = (150.0, 11.0, 0.05)

    taxiway_pos: Tuple[float, float, float] = (0.0, -75.0, 0.05)
    taxiway_half: Tuple[float, float, float] = (150.0, 8.0, 0.05)

    apron_pos: Tuple[float, float, float] = (0.0, -28.0, 0.05)
    apron_half: Tuple[float, float, float] = (150.0, 30.0, 0.05)

    # ---- Buildings -----------------------------------------------------
    terminal_pos: Tuple[float, float, float] = (0.0, 48.0, 6.0)
    terminal_half: Tuple[float, float, float] = (92.0, 16.0, 6.0)

    tower_pos: Tuple[float, float, float] = (-110.0, 46.0, 17.0)
    tower_radius: float = 4.0
    tower_height: float = 34.0

    # ---- Gates ---------------------------------------------------------
    # X positions of gate stands; Y of the parking line. Each gate holds one
    # parked aircraft (nose facing the apron, -Y) plus turnaround vehicles.
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
    agent_half: Tuple[float, float, float] = (1.6, 0.9, 0.5)
    agent_start: Tuple[float, float, float] = (-110.0, -42.0, 0.5)

    # ---- Ambient ground traffic (moving obstacles) ---------------------
    n_movers: int = 4
    mover_half: Tuple[float, float, float] = (1.8, 1.0, 0.5)
    mover_speed: float = 5.0  # m/s patrol speed

    # ---- Live taxiing aircraft on the parallel taxiway -----------------
    n_taxi_aircraft: int = 2
    taxi_aircraft_speed: float = 9.0
    taxi_y: float = -75.0        # taxiway centreline
    taxi_x_span: float = 135.0   # aircraft patrol between -span and +span

    # ---- Turnaround service vehicles at each gate ----------------------
    service_offset_x: float = 6.5   # parked beside the fuselage
    service_y: float = -14.0        # forward of the wing, clear of the engines

    # ---- Mission target points ----------------------------------------
    baggage_dock: Tuple[float, float] = (108.0, -28.0)  # approach point at baggage hall
    dock_offset: float = 13.0       # metres ahead of the nose for the dock mission
    patrol_route: Tuple[Tuple[float, float], ...] = (
        (-120.0, -47.0), (-60.0, -47.0), (0.0, -47.0), (60.0, -47.0), (120.0, -47.0),
    )

    # ---- Colours (rgba 0-1) -------------------------------------------
    col_ground: Tuple[float, float, float, float] = (0.20, 0.22, 0.24, 1.0)
    col_runway: Tuple[float, float, float, float] = (0.10, 0.10, 0.12, 1.0)
    col_taxiway: Tuple[float, float, float, float] = (0.16, 0.16, 0.18, 1.0)
    col_apron: Tuple[float, float, float, float] = (0.30, 0.31, 0.33, 1.0)
    col_terminal: Tuple[float, float, float, float] = (0.72, 0.76, 0.80, 1.0)
    col_tower: Tuple[float, float, float, float] = (0.85, 0.86, 0.88, 1.0)
    col_marking: Tuple[float, float, float, float] = (0.95, 0.95, 0.95, 1.0)
    col_aircraft: Tuple[float, float, float, float] = (0.90, 0.92, 0.95, 1.0)
    col_taxi_aircraft: Tuple[float, float, float, float] = (0.86, 0.89, 0.94, 1.0)
    col_agent: Tuple[float, float, float, float] = (0.10, 0.70, 0.95, 1.0)
    col_mover: Tuple[float, float, float, float] = (0.95, 0.45, 0.10, 1.0)
    col_baggage: Tuple[float, float, float, float] = (0.55, 0.42, 0.25, 1.0)
    col_fuel: Tuple[float, float, float, float] = (0.75, 0.20, 0.20, 1.0)
    col_fuel_truck: Tuple[float, float, float, float] = (0.90, 0.60, 0.15, 1.0)
    col_catering: Tuple[float, float, float, float] = (0.30, 0.65, 0.42, 1.0)


@dataclass
class EnvConfig:
    """Reinforcement-learning / simulation parameters."""

    # Simulation
    sim_dt: float = 0.01
    frame_skip: int = 5  # -> control step of 0.05 s
    max_steps: int = 800

    # Agent kinematics (unicycle model)
    max_speed: float = 12.0      # m/s forward
    max_reverse_speed: float = 3.0
    max_yaw_rate: float = 1.6    # rad/s

    # ---- Mission -------------------------------------------------------
    # One of: "navigate", "dock_aircraft", "shuttle", "patrol"
    task: str = "navigate"
    goal_radius: float = 4.0       # arrival tolerance for most waypoints
    dock_radius: float = 3.0       # tighter tolerance for the docking mission
    patrol_radius: float = 6.0     # looser tolerance for patrol waypoints
    spawn_radius: float = 6.0      # randomisation of the start pose

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
    w_waypoint: float = 8.0        # bonus for reaching an intermediate waypoint

    # Observation
    include_lidar: bool = True
    include_rgb: bool = False
    rgb_width: int = 160
    rgb_height: int = 90

    # Camera for the viewer / rgb render
    cam_distance: float = 26.0
    cam_elevation: float = -55.0
    cam_azimuth: float = -90.0
