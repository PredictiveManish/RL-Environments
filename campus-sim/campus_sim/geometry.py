"""Geometry helpers and the static campus world model (numpy-vectorised)."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "campus.json")


def point_in_polygon(x: float, y: float, poly) -> bool:
    """Scalar ray-casting point-in-polygon. `poly` is a sequence of [x, y]."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def _edges(poly: np.ndarray):
    x1 = poly[:, 0]
    y1 = poly[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)
    return x1, y1, x2, y2


def pip_np(x: float, y: float, poly: np.ndarray) -> bool:
    """Vectorised point-in-polygon for a single query point."""
    x1, y1, x2, y2 = _edges(poly)
    cond = (y1 > y) != (y2 > y)
    dy = y2 - y1
    dy = np.where(dy == 0, 1e-12, dy)
    xint = (x2 - x1) * (y - y1) / dy + x1
    return bool(np.count_nonzero(cond & (x < xint)) % 2)


def dist_edges_np(x: float, y: float, poly: np.ndarray) -> float:
    x1, y1, x2, y2 = _edges(poly)
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    L2 = np.where(L2 == 0, 1.0, L2)
    t = np.clip(((x - x1) * dx + (y - y1) * dy) / L2, 0.0, 1.0)
    return float(np.min(np.hypot(x - (x1 + t * dx), y - (y1 + t * dy))))


@dataclass
class Building:
    id: str
    name: str
    kind: str
    height: float
    polygon: np.ndarray          # (N, 2) float
    centroid: np.ndarray         # (2,)

    @property
    def radius(self) -> float:
        return float(np.max(np.linalg.norm(self.polygon - self.centroid, axis=1)))


@dataclass
class Campus:
    name: str
    meta: dict
    boundary: np.ndarray                  # (N, 2)
    buildings: list = field(default_factory=list)
    paths: list = field(default_factory=list)
    pois: list = field(default_factory=list)

    def __post_init__(self):
        self._bcent = np.array([b.centroid for b in self.buildings])   # (B, 2)
        self._brad = np.array([b.radius for b in self.buildings])      # (B,)

    # ---- extents --------------------------------------------------------
    @property
    def bbox(self):
        b = self.boundary
        return (float(b[:, 0].min()), float(b[:, 1].min()),
                float(b[:, 0].max()), float(b[:, 1].max()))

    @property
    def center(self) -> np.ndarray:
        return self.boundary.mean(axis=0)

    @property
    def scale(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return 0.5 * math.hypot(x1 - x0, y1 - y0)

    # ---- queries --------------------------------------------------------
    def inside_boundary(self, x, y) -> bool:
        return pip_np(x, y, self.boundary)

    def collides(self, x, y, radius: float = 0.0) -> bool:
        """True if a disc of `radius` at (x, y) hits a building or leaves campus."""
        if not pip_np(x, y, self.boundary):
            return True
        if radius <= 0.0:
            for b in self.buildings:
                if pip_np(x, y, b.polygon):
                    return True
            return False
        # quick reject by centroid distance, then exact test
        d = np.hypot(self._bcent[:, 0] - x, self._bcent[:, 1] - y)
        for i in np.nonzero(d < self._brad + radius + 1.0)[0]:
            poly = self.buildings[i].polygon
            if pip_np(x, y, poly) or dist_edges_np(x, y, poly) < radius:
                return True
        return False

    def nearest_building(self, x, y):
        d = np.hypot(self._bcent[:, 0] - x, self._bcent[:, 1] - y)
        i = int(np.argmin(d))
        return self.buildings[i], float(d[i])

    def to_lonlat(self, x, y):
        mx = 111320.0 * math.cos(math.radians(self.meta["origin_lat"]))
        my = 110574.0
        return (self.meta["origin_lon"] + x / mx, self.meta["origin_lat"] + y / my)


def load_campus(path: str | None = None) -> Campus:
    path = path or _DEFAULT
    with open(path) as f:
        raw = json.load(f)
    buildings = [Building(id=b["id"], name=b["name"], kind=b["kind"],
                          height=b["height"],
                          polygon=np.asarray(b["polygon"], dtype=float),
                          centroid=np.asarray(b["centroid"], dtype=float))
                 for b in raw["buildings"]]
    return Campus(name=raw["meta"]["name"], meta=raw["meta"],
                  boundary=np.asarray(raw["boundary"], dtype=float),
                  buildings=buildings, paths=raw["paths"], pois=raw["pois"])
