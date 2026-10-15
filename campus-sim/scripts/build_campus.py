"""
build_campus.py
---------------
Builds data/campus.json -- the static world model for the IKGPTU Main Campus
(Kapurthala) RL environment.

Provenance of the geometry
==========================
* CAMPUS BOUNDARY  -> REAL.  Taken from OpenStreetMap way 123246355
  ("I. K. Gujral Punjab Technical University", amenity=university), fetched
  through the OSM Nominatim API.  The polygon's area comes out at ~79 acres,
  which matches the university's stated main-campus size of 78.16 acres.
* AUDITORIUM POINT -> REAL.  "Sri Guru Nanak Dev Ji Auditorium" is a named OSM
  feature at 31.3536063 N, 75.4570056 E, i.e. local (x=-13, y=29) metres.
* BUILDING FOOTPRINTS -> ILLUSTRATIVE.  Individual campus buildings are not
  mapped in OpenStreetMap (only the auditorium is).  The footprints below are a
  documented, plausible layout placed *inside the real boundary*, sized and
  labelled after the facilities the university lists publicly (auditorium,
  central library / Knowledge Resource Centre, academic blocks, hostels,
  cafeteria, workshop, sports complex, guest house, main building).
  Run scripts/refresh_from_osm.py on a machine where Overpass is reachable to
  replace these with true footprints; the environment code does not change.

Coordinate frame
================
Local ENU metres, origin at the campus centroid.
    x  = east  (+ right)
    y  = north (+ up)
Every polygon is a list of [x, y] vertices in metres.
"""
from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ORIGIN_LON = 75.457137
ORIGIN_LAT = 31.353348

# Real OSM campus boundary, already projected to local metres (see module docstring).
BOUNDARY = [
    [-693, 39], [-372, -45], [-125, -108], [185, -191], [248, -206],
    [648, -364], [749, -166], [204, 23], [180, 26], [169, 27], [158, 30],
    [145, 35], [131, 44], [124, 51], [-8, 98], [-166, 157], [-357, 226],
    [-525, 286], [-693, 39],
]


def rect(cx, cy, w, d, rot_deg):
    """Axis-aligned rectangle (w x d) centred at (cx, cy), rotated by rot_deg."""
    a = math.radians(rot_deg)
    ca, sa = math.cos(a), math.sin(a)
    hw, hd = w / 2.0, d / 2.0
    corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    return [[round(cx + px * ca - py * sa, 1), round(cy + px * sa + py * ca, 1)]
            for px, py in corners]


def centroid(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [round(sum(xs) / len(xs), 1), round(sum(ys) / len(ys), 1)]


# --- Buildings -------------------------------------------------------------
# (id, name, kind, centre_x, centre_y, width, depth, rot, height_m)
ROT = -14.0  # campus long axis, WSW->ENE
BUILDINGS = [
    ("admin",      "Main Administrative Building",     "admin",     -230,  70, 55, 26, ROT, 18),
    ("library",    "Knowledge Resource Centre",        "library",   -120,  60, 45, 30, ROT, 20),
    ("auditorium", "Sri Guru Nanak Dev Ji Auditorium", "auditorium", -13,  29, 62, 40, ROT, 22),
    ("acad_a",     "Academic Block A (Engineering)",   "academic",  -430, 130, 70, 28, ROT, 16),
    ("acad_b",     "Academic Block B (Engineering)",   "academic",  -330,  60, 70, 28, ROT, 16),
    ("acad_c",     "Academic Block C (Management)",    "academic",  -520, 190, 55, 26, ROT, 16),
    ("hostel_m",   "Boys Hostel",                      "hostel",     110,  20, 50, 20, ROT, 15),
    ("hostel_f",   "Girls Hostel",                     "hostel",     180, -10, 50, 20, ROT, 15),
    ("cafeteria",  "Cafeteria & Mess",                 "cafeteria",  250, -40, 40, 25, ROT, 10),
    ("workshop",   "Workshop & Laboratories",          "workshop",   330, -60, 60, 30, ROT, 14),
    ("sports",     "Sports Complex",                   "sports",     420, -90, 70, 40, ROT, 12),
    ("guest",      "Guest House",                      "guest",      500, -120, 35, 22, ROT, 12),
]


def build():
    buildings = []
    for bid, name, kind, cx, cy, w, d, rot, h in BUILDINGS:
        poly = rect(cx, cy, w, d, rot)
        buildings.append({
            "id": bid, "name": name, "kind": kind, "height": h,
            "polygon": poly, "centroid": centroid(poly),
            "footprint_m2": round(w * d, 1),
        })

    # --- Roads / walkways -------------------------------------------------
    # A spine road running the length of the campus, plus a loop on each end.
    spine = [[-650, 165], [-430, 95], [-230, 40], [-13, 8], [180, -25],
             [330, -70], [500, -140], [700, -200]]
    north_walk = [[-520, 230], [-360, 190], [-170, 120], [-13, 60],
                  [130, 20], [300, -30]]
    south_walk = [[-330, -80], [-125, -150], [120, -180], [300, -230],
                  [500, -280]]
    paths = [
        {"name": "Campus Spine Road", "kind": "road", "points": spine},
        {"name": "North Walkway", "kind": "walk", "points": north_walk},
        {"name": "South Walkway", "kind": "walk", "points": south_walk},
    ]

    # Branch connectors: each building to its nearest spine vertex.
    def nearest_spine(c):
        best, bd = None, 1e18
        for p in spine:
            dd = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
            if dd < bd:
                bd, best = dd, p
        return best

    for b in buildings:
        c = b["centroid"]
        s = nearest_spine(c)
        paths.append({"name": f"Path to {b['name']}", "kind": "branch",
                      "points": [s, c]})

    # --- Points of interest (goal locations = building entrances) ----------
    pois = []
    for b in buildings:
        c = b["centroid"]
        s = nearest_spine(c)
        vx, vy = s[0] - c[0], s[1] - c[1]
        n = math.hypot(vx, vy) or 1.0
        # entrance sits a few metres in front of the building, toward the spine
        ex, ey = c[0] + vx / n * 28.0, c[1] + vy / n * 28.0
        pois.append({"name": b["name"], "kind": "entrance",
                     "xy": [round(ex, 1), round(ey, 1)], "building_id": b["id"]})

    campus = {
        "meta": {
            "name": "IKGPTU Main Campus, Kapurthala",
            "university": "I. K. Gujral Punjab Technical University",
            "location": "Jalandhar-Kapurthala Highway, VPO Ibban, Kapurthala 144603, Punjab, India",
            "origin_lon": ORIGIN_LON, "origin_lat": ORIGIN_LAT,
            "crs": "local ENU metres; x=east, y=north; origin at campus centroid",
            "boundary_source": "OpenStreetMap way 123246355 via Nominatim (ODbL)",
            "building_source": "illustrative layout inside the real boundary; see refresh_from_osm.py",
            "area_acres": 79.1,
        },
        "boundary": BOUNDARY,
        "buildings": buildings,
        "paths": paths,
        "pois": pois,
    }
    return campus


def main():
    campus = build()
    out = os.path.join(ROOT, "data", "campus.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(campus, f, indent=1)

    # sanity: every building centroid must be inside the boundary
    def pip(pt, poly):
        x, y = pt
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside

    bad = [b["name"] for b in campus["buildings"]
           if not pip(b["centroid"], campus["boundary"])]
    print(f"wrote {out}")
    print(f"buildings: {len(campus['buildings'])}, paths: {len(campus['paths'])}, "
          f"pois: {len(campus['pois'])}")
    print("outside boundary:", bad if bad else "none")


if __name__ == "__main__":
    main()
