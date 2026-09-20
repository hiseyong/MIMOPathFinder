"""Grid-size-agnostic RF/MIMO physics and pathfinding utilities.

Successor to simple_2D/RF_source_seeking_2D.py's channel model, generalised
to operate on any occupancy grid shape (world.shape) instead of a single
hardcoded (NX, NY) — needed because each ADWA benchmark building has its own
map size.

Grid convention: a point is (x, y); world[y, x] is True for a wall.
"""
from __future__ import annotations

import heapq
from collections import deque
from math import pi

import numpy as np

# Ideal narrowband MIMO configuration -- unchanged from the simple_2D phase.
N_TX = 4
N_RX = 4
CARRIER_HZ = 2.4e9
LIGHT_SPEED = 299_792_458.0
WAVELENGTH = LIGHT_SPEED / CARRIER_HZ
ELEMENT_SPACING = WAVELENGTH / 2
TX_POWER_W = 0.1             # Total transmit power: 20 dBm
NOISE_POWER_W = 1e-12        # Receiver noise power: -90 dBm


def neighbours(point: tuple[int, int], world: np.ndarray):
    ny, nx = world.shape
    x, y = point
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        px, py = x + dx, y + dy
        if 0 <= px < nx and 0 <= py < ny and not world[py, px]:
            yield px, py


def astar(start: tuple[int, int], goal: tuple[int, int], world: np.ndarray):
    """Return a shortest collision-free four-neighbour path."""
    frontier = [(0, start)]
    came_from = {start: None}
    cost = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in neighbours(current, world):
            new_cost = cost[current] + 1
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                priority = new_cost + abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current
    else:
        raise RuntimeError("The goal is unreachable")

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    return path[::-1]


def bfs_distance_field(world: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """Wall-aware shortest step-count from every free cell to ``source``.

    A single BFS flood fill over the four-neighbour free-cell graph, so a
    cell's value is the length of an actually reachable route around walls,
    never Euclidean/straight-line distance. ``world``/``source`` are fixed
    for the life of an episode's map, so callers should compute this once
    per map and reuse it instead of re-running BFS on every step.
    """
    dist = np.full(world.shape, -1, dtype=np.int32)
    sx, sy = source
    dist[sy, sx] = 0
    frontier = deque([source])
    while frontier:
        current = frontier.popleft()
        cx, cy = current
        for nx, ny in neighbours(current, world):
            if dist[ny, nx] == -1:
                dist[ny, nx] = dist[cy, cx] + 1
                frontier.append((nx, ny))
    return dist


def bresenham(a: tuple[int, int], b: tuple[int, int]):
    """Yield cells sampled along the straight line from a to b."""
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    while True:
        yield x0, y0
        if (x0, y0) == (x1, y1):
            return
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += sx
        if twice_error <= dx:
            error += dx
            y0 += sy


def count_path_corners(path: list[tuple[int, int]]) -> int:
    """Number of direction changes along a four-neighbour path."""
    if len(path) < 3:
        return 0
    corners = 0
    prev = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for (x0, y0), (x1, y1) in zip(path[1:-1], path[2:]):
        cur = (x1 - x0, y1 - y0)
        corners += cur != prev
        prev = cur
    return corners


def ula_positions(center: tuple[float, float], n_elements: int, heading: float):
    """Element locations for an ideal ULA, centred on a 2D platform."""
    perpendicular = (-np.sin(heading), np.cos(heading))
    offsets = (np.arange(n_elements) - (n_elements - 1) / 2) * ELEMENT_SPACING
    return np.array([
        (center[0] + offset * perpendicular[0], center[1] + offset * perpendicular[1])
        for offset in offsets
    ])


def mimo_channel(
    robot: tuple[float, float], source: tuple[float, float], world: np.ndarray,
    rx_heading: float | None = None,
) -> np.ndarray:
    """Return the ideal 4x4 narrowband baseband channel matrix H.

    Each transmitter--receiver pair receives a complex coefficient with
    free-space amplitude decay, carrier phase, and 18 dB attenuation per wall
    cell crossed. ``robot``/``source`` are grid coordinates in the same units
    as ``world`` (i.e. already in cells, not metres -- callers on a
    downsampled map must convert first).
    """
    ny, nx = world.shape
    direction = (np.arctan2(source[1] - robot[1], source[0] - robot[0])
                 if rx_heading is None else rx_heading)
    tx = ula_positions(source, N_TX, heading=0.0)
    rx = ula_positions(robot, N_RX, heading=direction)
    h = np.zeros((N_RX, N_TX), dtype=np.complex128)

    for r, r_pos in enumerate(rx):
        for t, t_pos in enumerate(tx):
            # One grid cell is the near-field resolution of this map; the cap
            # prevents a co-located cell from dominating the amplitude.
            distance = max(np.linalg.norm(r_pos - t_pos), 1.0)
            endpoints = (
                (int(round(t_pos[0])), int(round(t_pos[1]))),
                (int(round(r_pos[0])), int(round(r_pos[1]))),
            )
            wall_hits = sum(
                world[y, x]
                for x, y in bresenham(*endpoints)
                if 0 <= x < nx and 0 <= y < ny
            )
            amplitude = (WAVELENGTH / (4 * pi * distance)) * 10 ** (-18 * wall_hits / 20)
            h[r, t] = amplitude * np.exp(-1j * 2 * pi * distance / WAVELENGTH)
    return h


def mimo_metrics(h: np.ndarray) -> tuple[float, float]:
    """Return mean received power (dB, unit transmit power) and channel rank."""
    mean_power_db = 10 * np.log10(np.mean(np.abs(h) ** 2) + 1e-30)
    rank = np.linalg.matrix_rank(h, tol=1e-12)
    return mean_power_db, rank
