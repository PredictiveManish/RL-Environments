"""Ground-vehicle models: the RL agent and the ambient traffic movers.

The agent is a box chassis with a free joint. Its motion is driven by a
*unicycle* velocity controller in the environment (see ``airport_env.py``),
which keeps navigation stable and easy to train while still producing real
MuJoCo contacts for collision detection.
"""

from __future__ import annotations

from typing import Sequence


def agent_xml(
    name: str,
    pos: Sequence[float],
    half: Sequence[float],
    colour: Sequence[float],
) -> str:
    """The vehicle the RL policy controls. Nose faces +X."""
    hx, hy, hz = half
    r, g, b, a = colour
    return f"""
    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
      <freejoint name="{name}_joint"/>
      <geom name="{name}_chassis" type="box" size="{hx} {hy} {hz}"
            rgba="{r} {g} {b} {a}" contype="1" conaffinity="1"/>
      <geom name="{name}_cab" type="box" pos="{hx*0.35} 0 {hz*0.8}"
            size="{hx*0.45} {hy*0.8} {hz*0.7}" rgba="0.15 0.15 0.18 1"/>
      <geom name="{name}_beacon" type="sphere" pos="{-hx*0.7} 0 {hz*1.3}"
            size="0.22" rgba="1 0.75 0 1"/>
    </body>"""


def mover_xml(
    name: str,
    pos: Sequence[float],
    half: Sequence[float],
    colour: Sequence[float],
) -> str:
    """Ambient ground traffic - a dynamic obstacle the agent must avoid."""
    hx, hy, hz = half
    r, g, b, a = colour
    return f"""
    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
      <freejoint name="{name}_joint"/>
      <geom name="{name}_chassis" type="box" size="{hx} {hy} {hz}"
            rgba="{r} {g} {b} {a}"/>
      <geom name="{name}_cab" type="box" pos="{hx*0.3} 0 {hz*0.8}"
            size="{hx*0.4} {hy*0.75} {hz*0.6}" rgba="0.15 0.15 0.18 1"/>
    </body>"""
