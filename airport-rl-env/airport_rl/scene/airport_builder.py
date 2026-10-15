"""Builds the complete airport scene as a MuJoCo MJCF document.

Everything is generated procedurally from primitives, so there are no external
mesh files to download. ``AirportScene.build()`` returns the XML string plus a
metadata dict the environment uses (gate positions, mover patrol routes, the
names of the drivable surfaces, etc.).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .aircraft_factory import aircraft_xml
from .vehicle_factory import agent_xml, mover_xml


def _f(v) -> str:
    return " ".join(f"{x:.4f}" for x in v)


class AirportScene:
    def __init__(self, cfg):
        self.cfg = cfg

    # ------------------------------------------------------------------
    def build(self) -> Tuple[str, Dict]:
        cfg = self.cfg
        body_parts: List[str] = []

        # ---- Ground + movement surfaces (all "drivable") ---------------
        body_parts.append(
            f'<geom name="drive_ground" type="plane" size="420 420 0.1" '
            f'material="grid" contype="0" conaffinity="0"/>'
        )
        body_parts.append(self._slab("drive_runway", cfg.runway_pos, cfg.runway_half, cfg.col_runway, solid=False))
        body_parts.append(self._slab("drive_taxiway", cfg.taxiway_pos, cfg.taxiway_half, cfg.col_taxiway, solid=False))
        body_parts.append(self._slab("drive_apron", cfg.apron_pos, cfg.apron_half, cfg.col_apron, solid=False))

        # ---- Runway markings (thin white strips, not drivable) ---------
        body_parts += self._runway_markings()

        # ---- Buildings -------------------------------------------------
        body_parts.append(self._slab("terminal", cfg.terminal_pos, cfg.terminal_half, cfg.col_terminal))
        # Terminal roof lip for a little silhouette
        tp = cfg.terminal_pos
        th = cfg.terminal_half
        body_parts.append(
            f'<geom name="terminal_roof" type="box" contype="0" conaffinity="0" '
            f'pos="{tp[0]} {tp[1]} {tp[2] + th[2] + 0.4}" '
            f'size="{th[0] + 1.5} {th[1] + 1.5} 0.4" rgba="0.55 0.60 0.66 1"/>'
        )
        # Glazed landside/apron facade band for a bit of realism
        body_parts.append(
            f'<geom name="terminal_glass" type="box" contype="0" conaffinity="0" '
            f'pos="{tp[0]} {tp[1] - th[1] - 0.15} {tp[2] * 0.85}" '
            f'size="{th[0] * 0.99} 0.2 {th[2] * 0.55}" rgba="0.20 0.34 0.46 1"/>'
        )
        # Control tower: column + cabin
        tw = cfg.tower_pos
        body_parts.append(
            f'<geom name="tower_column" type="cylinder" '
            f'pos="{tw[0]} {tw[1]} {cfg.tower_height * 0.5}" '
            f'size="{cfg.tower_radius} {cfg.tower_height * 0.5}" rgba="{_f(cfg.col_tower)}"/>'
        )
        body_parts.append(
            f'<geom name="tower_cabin" type="cylinder" '
            f'pos="{tw[0]} {tw[1]} {cfg.tower_height + 2.0}" '
            f'size="{cfg.tower_radius + 2.2} 2.0" rgba="0.20 0.30 0.40 1"/>'
        )

        # ---- Airside service facilities --------------------------------
        body_parts.append(self._slab("baggage_hall", cfg.baggage_pos, cfg.baggage_half, cfg.col_baggage))
        fp = cfg.fuel_pos
        body_parts.append(
            f'<geom name="fuel_tank_1" type="cylinder" pos="{fp[0]} {fp[1] + 8} {cfg.fuel_height*0.5}" '
            f'size="{cfg.fuel_radius} {cfg.fuel_height*0.5}" rgba="{_f(cfg.col_fuel)}"/>'
        )
        body_parts.append(
            f'<geom name="fuel_tank_2" type="cylinder" pos="{fp[0]} {fp[1] - 8} {cfg.fuel_height*0.5}" '
            f'size="{cfg.fuel_radius} {cfg.fuel_height*0.5}" rgba="{_f(cfg.col_fuel)}"/>'
        )
        body_parts.append(self._slab("maintenance_hangar", cfg.maintenance_pos, cfg.maintenance_half,
                                     (0.60, 0.62, 0.66, 1.0)))

        # ---- Gates: stand marking + a parked aircraft each -------------
        gate_positions: List[Tuple[float, float]] = []
        for i, gx in enumerate(cfg.gate_xs):
            # Stand marking just in front of the aircraft (toward the apron)
            stand_y = cfg.gate_y - 9.0
            body_parts.append(
                f'<geom name="gate_{i}_mark" type="box" contype="0" conaffinity="0" '
                f'pos="{gx} {stand_y} 0.06" size="6.0 5.0 0.02" rgba="0.9 0.9 0.2 1"/>'
            )
            body_parts.append(
                aircraft_xml(f"aircraft_{i}", (gx, cfg.gate_y, cfg.aircraft_z), yaw=-1.5708,
                             colour=cfg.col_aircraft)
            )
            gate_positions.append((gx, stand_y))

        # ---- Agent -----------------------------------------------------
        body_parts.append(agent_xml("agent", cfg.agent_start, cfg.agent_half, cfg.col_agent))

        # ---- Ambient traffic (moving obstacles) ------------------------
        mover_waypoints = self._mover_routes()
        for i, (p0, _p1) in enumerate(mover_waypoints):
            body_parts.append(
                mover_xml(f"mover_{i}", (p0[0], p0[1], cfg.mover_half[2]), cfg.mover_half, cfg.col_mover)
            )

        xml = self._wrap_xml("\n".join(body_parts))

        meta = {
            "gate_positions": gate_positions,
            "mover_waypoints": mover_waypoints,
            "driveable_prefix": "drive_",
        }
        return xml, meta

    # ------------------------------------------------------------------
    def _slab(self, name, pos, half, colour, solid: bool = True) -> str:
        cc = ' contype="0" conaffinity="0"' if not solid else ""
        return (
            f'<geom name="{name}" type="box"{cc} pos="{_f(pos)}" '
            f'size="{_f(half)}" rgba="{_f(colour)}"/>'
        )


    def _runway_markings(self) -> List[str]:
        cfg = self.cfg
        rx, ry, _ = cfg.runway_pos
        half_len = cfg.runway_half[0]
        parts = []
        # Dashed centreline
        n = 14
        for i in range(n):
            x = -half_len + (2 * half_len) * (i + 0.5) / n
            parts.append(
                f'<geom name="rwy_centre_{i}" type="box" contype="0" conaffinity="0" '
                f'pos="{x:.3f} {ry} 0.11" size="4.0 0.5 0.02" rgba="{_f(cfg.col_marking)}"/>'
            )
        # Threshold bars at both ends
        for k, sx in enumerate((-half_len + 4.0, half_len - 4.0)):
            for j, oy in enumerate((-5.0, -2.0, 2.0, 5.0)):
                parts.append(
                    f'<geom name="rwy_thr_{k}_{j}" type="box" contype="0" conaffinity="0" '
                    f'pos="{sx:.3f} {ry + oy} 0.11" size="1.0 0.7 0.02" rgba="{_f(cfg.col_marking)}"/>'
                )
        # Taxiway centreline
        ty = cfg.taxiway_pos[1]
        for i in range(n):
            x = -half_len + (2 * half_len) * (i + 0.5) / n
            parts.append(
                f'<geom name="twy_centre_{i}" type="box" contype="0" conaffinity="0" '
                f'pos="{x:.3f} {ty} 0.11" size="4.0 0.4 0.02" rgba="0.85 0.75 0.1 1"/>'
            )
        return parts

    def _mover_routes(self):
        """Deterministic patrol segments for the ambient traffic."""
        # Patrol lanes stay in the open apron *in front of* the aircraft
        # (y <= -20, clear of the stands at y in [-16, 0]), and clear of the
        # fuel depot (x < -121), baggage hall (x > 114) and maintenance
        # hangar (y < -50).
        routes = [
            ((-100.0, -22.0), (100.0, -22.0)),
            ((100.0, -32.0), (-100.0, -32.0)),
            ((-90.0, -42.0), (90.0, -42.0)),
            ((95.0, -26.0), (-95.0, -26.0)),
            ((-80.0, -38.0), (80.0, -38.0)),
            ((85.0, -20.0), (-85.0, -20.0)),
        ]
        return routes[: self.cfg.n_movers]

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_xml(body: str) -> str:
        return f"""
<mujoco model="airport">
  <compiler angle="radian" coordinate="local" autolimits="true"/>
  <option timestep="0.01" gravity="0 0 0" integrator="Euler"/>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.35 0.35 0.35" specular="0.15 0.15 0.15"/>
    <rgba haze="0.12 0.20 0.30 1"/>
    <global azimuth="-140" elevation="-22" offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.18 0.20 0.22" rgb2="0.22 0.24 0.26"/>
    <material name="grid" texture="grid" texrepeat="40 40" reflectance="0.08"/>
  </asset>

  <worldbody>
    <light pos="0 0 160" dir="0 0 -1" directional="true"
           diffuse="0.85 0.85 0.85" ambient="0.35 0.35 0.35"/>
    <light pos="-150 -120 60" dir="0.6 0.5 -1" directional="true"
           diffuse="0.35 0.35 0.4" ambient="0.1 0.1 0.12"/>
{body}
  </worldbody>
</mujoco>
"""
