"""
refresh_from_osm.py
-------------------
Replace the illustrative building footprints with REAL OpenStreetMap footprints.

The sandbox this project was assembled in cannot reach the Overpass API, so the
shipped data/campus.json uses a documented illustrative layout inside the real
campus boundary. Run this script on any normal machine / network and the
footprints become real OSM data. The environment code does not change -- it only
reads data/campus.json.

    pip install requests
    python scripts/refresh_from_osm.py --apply        # overwrite data/campus.json
    python scripts/refresh_from_osm.py                # write data/campus_osm.json instead

Data (c) OpenStreetMap contributors, ODbL 1.0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Real campus boundary + frame, from scripts/build_campus.py
ORIGIN_LON = 75.457137
ORIGIN_LAT = 31.353348
BBOX = (31.3460, 75.4420, 31.3600, 75.4720)  # south, west, north, east
MX = 111320.0 * math.cos(math.radians(ORIGIN_LAT))
MY = 110574.0

OVERPASS = "https://overpass-api.de/api/interpreter"


def to_local(lon, lat):
    return [round((lon - ORIGIN_LON) * MX, 1), round((lat - ORIGIN_LAT) * MY, 1)]


def fetch():
    s, w, n, e = BBOX
    q = f"""[out:json][timeout:120];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
);
out geom;"""
    data = urllib.parse.urlencode({"data": q}).encode()
    req = urllib.request.Request(OVERPASS, data=data,
                                 headers={"User-Agent": "CampusSim/1.0 (student project)"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def height_of(tags):
    if "height" in tags:
        try:
            return float(str(tags["height"]).replace("m", "").strip())
        except ValueError:
            pass
    if "building:levels" in tags:
        try:
            return float(tags["building:levels"]) * 3.2
        except ValueError:
            pass
    return 12.0


def build(raw):
    with open(os.path.join(ROOT, "data", "campus.json")) as f:
        campus = json.load(f)  # keep boundary / paths / meta, swap buildings

    buildings = []
    for el in raw.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        poly = [to_local(g["lon"], g["lat"]) for g in el["geometry"]]
        if len(poly) < 4:
            continue
        tags = el.get("tags", {})
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        buildings.append({
            "id": f"osm{el['id']}",
            "name": tags.get("name", f"Building {el['id']}"),
            "kind": tags.get("building", "building"),
            "height": height_of(tags),
            "polygon": poly,
            "centroid": [round(cx, 1), round(cy, 1)],
            "osm_id": el["id"],
        })

    # entrances: front of each building toward the campus centre
    pois = []
    for b in buildings:
        c = b["centroid"]
        vx, vy = -c[0], -c[1]
        nn = math.hypot(vx, vy) or 1.0
        pois.append({"name": b["name"], "kind": "entrance",
                     "xy": [round(c[0] + vx / nn * 28, 1),
                            round(c[1] + vy / nn * 28, 1)],
                     "building_id": b["id"]})

    campus["buildings"] = buildings
    campus["pois"] = pois
    campus["meta"]["building_source"] = "OpenStreetMap Overpass (ODbL)"
    return campus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="overwrite data/campus.json (default: data/campus_osm.json)")
    args = ap.parse_args()

    print("querying Overpass...")
    raw = fetch()
    campus = build(raw)
    out = os.path.join(ROOT, "data",
                       "campus.json" if args.apply else "campus_osm.json")
    with open(out, "w") as f:
        json.dump(campus, f, indent=1)
    print(f"wrote {out}: {len(campus['buildings'])} real building footprints")
    print("re-run scripts/build_campus.py to regenerate paths if you changed the frame")


if __name__ == "__main__":
    main()
