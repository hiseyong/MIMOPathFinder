"""Load ADWA IROS-2020 benchmark building maps and navigation tasks.

Dataset: "Adaptive Dynamic Window Approach for Local Navigation" (IROS 2020),
Dobrevski & Skocaj, University of Ljubljana -- 17 real building floorplans as
ROS map_server-style PNG occupancy grids (1 cm/pixel) plus recorded
navigation tasks (start/goal pairs within 4 m) and multi-waypoint episodes.
Expected to be extracted at ./adwa_benchmark relative to the project root
(one *.png + *.yaml + *_tasks.txt + *_episodes.txt per building).

Grid convention matches rf_physics: a point is (x, y); world[y, x] is True
for a wall. Unlike simple_2D's fixed 1-unit-per-metre toy grid, each
building here is downsampled to a configurable physical cell size so BFS/A*
stay cheap regardless of a map's native pixel resolution.
"""
from __future__ import annotations

import ast
import glob
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adwa_benchmark")


def _read_flat_yaml(path: str) -> dict:
    """Minimal ``key: value`` parser for the benchmark's flat ROS map_server
    yaml files (no dependency on PyYAML for six scalar/list lines)."""
    result = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            value = value.strip()
            try:
                result[key.strip()] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                result[key.strip()] = value
    return result


@dataclass(frozen=True)
class BuildingMap:
    name: str
    world: np.ndarray        # bool grid, world[y, x] True == wall; downsampled resolution
    resolution_m: float      # metres per grid cell, after downsampling
    origin_m: tuple[float, float]  # world (metres) coordinate of pixel (0, 0)'s bottom-left corner


def list_buildings(data_dir: str = DEFAULT_DATA_DIR) -> list[str]:
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(data_dir, "*.png")))


def load_map(name: str, data_dir: str = DEFAULT_DATA_DIR, cell_size_m: float = 0.2) -> BuildingMap:
    """Load one building's occupancy grid, downsampled to ~cell_size_m per cell.

    ROS map_server convention: pixel value near 0 (black) is occupied, near 1
    (white) is free; image row 0 is the TOP of the map, but the yaml
    ``origin`` names the metric coordinate of the BOTTOM-LEFT pixel, so row
    index must be flipped when converting to/from world metres.
    """
    meta = _read_flat_yaml(os.path.join(data_dir, f"{name}.yaml"))
    native_res = float(meta["resolution"])
    origin_m = (float(meta["origin"][0]), float(meta["origin"][1]))
    occupied_thresh = float(meta.get("occupied_thresh", 0.65))

    img = plt.imread(os.path.join(data_dir, meta["image"]))
    if img.ndim == 3:
        img = img[..., 0]
    occupied = img < (1.0 - occupied_thresh)  # ROS: value = 1 - occupancy_probability

    factor = max(1, round(cell_size_m / native_res))
    h, w = occupied.shape
    h_trim, w_trim = h - h % factor, w - w % factor
    blocks = occupied[:h_trim, :w_trim].reshape(h_trim // factor, factor, w_trim // factor, factor)
    # Max-pool (any occupied pixel in the block -> wall cell): conservative,
    # never opens a gap in a wall that downsampling coordinates could clip.
    coarse_occupied = blocks.max(axis=(1, 3))
    # Row 0 of the image is the top of the map; flip so world[0, :] is the
    # bottom row, matching the yaml origin's bottom-left convention.
    world = coarse_occupied[::-1, :].copy()
    return BuildingMap(name=name, world=world, resolution_m=native_res * factor, origin_m=origin_m)


def meters_to_cell(building: BuildingMap, xy_m: tuple[float, float]) -> tuple[int, int]:
    x = int(round((xy_m[0] - building.origin_m[0]) / building.resolution_m))
    y = int(round((xy_m[1] - building.origin_m[1]) / building.resolution_m))
    ny, nx = building.world.shape
    return int(np.clip(x, 0, nx - 1)), int(np.clip(y, 0, ny - 1))


def load_tasks(name: str, data_dir: str = DEFAULT_DATA_DIR) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return the benchmark's recorded (start_m, goal_m) pairs for a building."""
    path = os.path.join(data_dir, f"{name}_tasks.txt")
    with open(path) as f:
        starts = ast.literal_eval(f.readline().strip())
        goals = ast.literal_eval(f.readline().strip())
    return list(zip(starts, goals))
