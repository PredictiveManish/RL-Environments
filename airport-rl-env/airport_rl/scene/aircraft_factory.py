"""Procedural aircraft models (pure MJCF, no external assets).

Each aircraft is a simplified narrow-body jet: fuselage, wings, tail and two
engines. Everything is generated from primitives so the project has *zero*
external model dependencies - it runs straight after ``pip install``.

Aircraft are static by default (parked at a gate). Pass ``free=True`` to add a
free joint so the aircraft can be driven along the taxiway by the environment.
"""

from __future__ import annotations

from typing import Sequence


def aircraft_xml(
    name: str,
    pos: Sequence[float],
    yaw: float = 0.0,
    colour: Sequence[float] = (0.9, 0.92, 0.95, 1.0),
    accent: Sequence[float] = (0.15, 0.35, 0.65, 1.0),
    free: bool = False,
) -> str:
    """Return the MJCF ``<body>`` for one aircraft, nose facing +X (before yaw)."""
    r, g, b, a = colour
    ar, ag, ab, aa = accent
    joint = f'\n      <freejoint name="{name}_joint"/>' if free else ""
    return f"""
    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" euler="0 0 {yaw}">{joint}
      <geom name="{name}_fuselage" type="capsule" fromto="-8 0 0 8 0 0"
            size="1.6" rgba="{r} {g} {b} {a}"/>
      <geom name="{name}_nose" type="sphere" pos="8 0 0" size="1.6"
            rgba="{r} {g} {b} {a}"/>
      <geom name="{name}_wing" type="box" pos="-0.5 0 -0.6"
            size="2.2 9.5 0.22" rgba="{r} {g} {b} {a}"/>
      <geom name="{name}_wingtip_l" type="box" pos="-0.5 9.0 0.25"
            size="1.0 0.35 0.9" rgba="{ar} {ag} {ab} {aa}"/>
      <geom name="{name}_wingtip_r" type="box" pos="-0.5 -9.0 0.25"
            size="1.0 0.35 0.9" rgba="{ar} {ag} {ab} {aa}"/>
      <geom name="{name}_tailfin" type="box" pos="-7.5 0 2.6"
            size="1.2 0.2 2.4" rgba="{ar} {ag} {ab} {aa}"/>
      <geom name="{name}_hstab" type="box" pos="-7.5 0 0.6"
            size="1.2 4.2 0.18" rgba="{r} {g} {b} {a}"/>
      <geom name="{name}_engine_l" type="cylinder" pos="1.5 5.2 -1.4"
            euler="1.5708 0 0" size="0.9 1.8" rgba="0.25 0.25 0.28 1"/>
      <geom name="{name}_engine_r" type="cylinder" pos="1.5 -5.2 -1.4"
            euler="1.5708 0 0" size="0.9 1.8" rgba="0.25 0.25 0.28 1"/>
    </body>"""
