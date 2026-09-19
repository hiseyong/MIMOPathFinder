"""2D RF source-seeking environment with a wall-connected U-shaped partition.

Run with:
    python rf_source_seeking_2d.py

Dependencies: numpy, matplotlib
"""

from __future__ import annotations

import heapq
from collections import deque
from math import pi

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


# Grid convention: a point is (x, y); world[y, x] is True for a wall.
NX, NY = 42, 30
AGENT = (6, 25)
SOURCE = (20, 11)

# Ideal narrowband MIMO configuration. Four elements per side is a compact,
# useful baseline before investigating larger arrays or non-ideal hardware.
N_TX = 4
N_RX = 4
CARRIER_HZ = 2.4e9
LIGHT_SPEED = 299_792_458.0
WAVELENGTH = LIGHT_SPEED / CARRIER_HZ
ELEMENT_SPACING = WAVELENGTH / 2
TX_POWER_W = 0.1             # Total transmit power: 20 dBm
NOISE_POWER_W = 1e-12        # Receiver noise power: -90 dBm


def make_world() -> np.ndarray:
    """Make a room whose partition is attached to the top wall, not an island.

    The connected partition is shaped like a U: its left leg starts at the top
    room wall, its base runs to the right, and its right leg points upward.
    Its only mouth is above the right leg, forcing a route around that opening.
    """
    world = np.zeros((NY, NX), dtype=bool)

    # Outer room walls.
    world[0, :] = world[-1, :] = True
    world[:, 0] = world[:, -1] = True

    # U-shaped, wall-connected divider: never an isolated central obstacle.
    world[1:19, 12] = True      # left leg: connected to the top wall
    world[18, 12:32] = True     # base
    world[5:19, 31] = True      # right leg; leaves one entrance above it
    return world


def neighbours(point: tuple[int, int], world: np.ndarray):
    x, y = point
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < NX and 0 <= ny < NY and not world[ny, nx]:
            yield nx, ny


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
        raise RuntimeError("The source is unreachable")

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
    for the life of the environment, so callers should compute this once and
    reuse it instead of re-running A*/BFS on every step.
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
    """Number of direction changes along a four-neighbour path.

    Each corner is a point where the shortest route can no longer continue in
    a straight line -- a proxy for how much the route must detour around
    obstacles that block direct line of sight to the goal.
    """
    if len(path) < 3:
        return 0
    corners = 0
    prev = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for (x0, y0), (x1, y1) in zip(path[1:-1], path[2:]):
        cur = (x1 - x0, y1 - y0)
        corners += cur != prev
        prev = cur
    return corners


def generate_random_world(rng: np.random.Generator, nx: int = NX, ny: int = NY,
                          max_attempts: int = 2000):
    """A random wall-connected single-partition room, source and start band.

    Kept deliberately "moderate": rejection-sampled so that, from a sample of
    candidate start cells, the direct line to the source is blocked by at
    most 2 wall cells and the shortest wall-aware route bends around at most
    3 corners. This gives varied layouts/source placements for training
    generalisation without generating unsolvable-looking mazes.
    """
    for _ in range(max_attempts):
        world = np.zeros((ny, nx), dtype=bool)
        world[0, :] = world[-1, :] = True
        world[:, 0] = world[:, -1] = True

        shape = rng.choice(("u", "l", "i"))
        left_x = int(rng.integers(4, nx // 2))
        right_x = int(rng.integers(nx // 2, nx - 4))
        base_y = int(rng.integers(ny // 3, int(ny * 0.7)))
        left_len = int(rng.integers(max(2, base_y // 2), base_y))
        right_len = int(rng.integers(max(2, base_y // 2), base_y))

        world[1:1 + left_len, left_x] = True
        world[base_y, left_x:right_x + 1] = True
        if shape in ("u", "l"):
            world[max(1, base_y - right_len):base_y, right_x] = True

        source_band = [(x, y) for y in range(1, max(2, base_y - 3))
                       for x in range(1, nx - 1) if not world[y, x]]
        start_band = [(x, y) for y in range(min(ny - 2, base_y + 3), ny - 1)
                      for x in range(1, nx - 1) if not world[y, x]]
        if len(source_band) < 10 or len(start_band) < 10:
            continue

        source = source_band[int(rng.integers(len(source_band)))]
        sample_size = min(8, len(start_band))
        sample_starts = [start_band[i] for i in rng.integers(0, len(start_band), size=sample_size)]

        constraints_ok = True
        for start in sample_starts:
            wall_hits = sum(world[y, x] for x, y in bresenham(start, source)
                            if 0 <= x < nx and 0 <= y < ny)
            if wall_hits > 2:
                constraints_ok = False
                break
            try:
                path = astar(start, source, world)
            except RuntimeError:
                constraints_ok = False
                break
            if count_path_corners(path) > 3:
                constraints_ok = False
                break
        if constraints_ok:
            return world, source, start_band
    raise RuntimeError("Could not generate a map satisfying the complexity constraints")


def received_power_map(world: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """Map mean 4x4 MIMO received power at every possible robot location."""
    power = np.full((NY, NX), np.nan)
    for y in range(NY):
        for x in range(NX):
            if world[y, x]:
                continue
            h = mimo_channel((x, y), source, world)
            power[y, x], _ = mimo_metrics(h)
    return power


def ula_positions(center: tuple[float, float], n_elements: int, heading: float):
    """Element locations for an ideal ULA, centred on a 2D platform.

    ``heading`` points along platform motion; the antenna line is perpendicular
    to it.  The RF source array has heading 0 and the robot array is rotated to
    face the source at each sampled robot position.
    """
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
    cell crossed. This deliberately simple model is a foundation for adding
    reflection paths and noise in the next experiment.
    """
    direction = (np.arctan2(source[1] - robot[1], source[0] - robot[0])
                 if rx_heading is None else rx_heading)
    tx = ula_positions(source, N_TX, heading=0.0)
    rx = ula_positions(robot, N_RX, heading=direction)
    h = np.zeros((N_RX, N_TX), dtype=np.complex128)

    for r, r_pos in enumerate(rx):
        for t, t_pos in enumerate(tx):
            # One grid cell is the near-field resolution of this map; the cap
            # prevents a co-located display cell from dominating the color scale.
            distance = max(np.linalg.norm(r_pos - t_pos), 1.0)
            # Grid points are rounded only to test whether the ray crosses a wall.
            endpoints = (
                (int(round(t_pos[0])), int(round(t_pos[1]))),
                (int(round(r_pos[0])), int(round(r_pos[1]))),
            )
            wall_hits = sum(
                world[y, x]
                for x, y in bresenham(*endpoints)
                if 0 <= x < NX and 0 <= y < NY
            )
            amplitude = (WAVELENGTH / (4 * pi * distance)) * 10 ** (-18 * wall_hits / 20)
            h[r, t] = amplitude * np.exp(-1j * 2 * pi * distance / WAVELENGTH)
    return h


def mimo_metrics(h: np.ndarray) -> tuple[float, float]:
    """Return mean received power (dB, unit transmit power) and channel rank."""
    mean_power_db = 10 * np.log10(np.mean(np.abs(h) ** 2) + 1e-30)
    rank = np.linalg.matrix_rank(h, tol=1e-12)
    return mean_power_db, rank


def transmit_mimo_pilot(
    h: np.ndarray, rng: np.random.Generator, noise_power: float = NOISE_POWER_W
):
    """Send one unit-power QPSK pilot vector and return x (Tx) and y (Rx)."""
    bits = rng.integers(0, 2, size=(N_TX, 2))
    x = ((2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)) / np.sqrt(2)
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(N_RX) + 1j * rng.standard_normal(N_RX)
    )
    return x, h @ x + noise


def db(value: np.ndarray | float) -> np.ndarray | float:
    """Convert a linear power quantity to dB without taking log(0)."""
    return 10 * np.log10(np.maximum(value, 1e-30))


def channel_report(
    h: np.ndarray, robot: tuple[int, int], source: tuple[int, int], world: np.ndarray
) -> None:
    """Print a broad set of interpretable narrowband MIMO channel features."""
    dx, dy = robot[0] - source[0], robot[1] - source[1]
    range_m = np.hypot(dx, dy)  # This prototype uses one grid cell as one metre.
    aod_deg = np.degrees(np.arctan2(dy, dx)) % 360
    aoa_deg = (aod_deg + 180) % 360
    centre_ray = list(bresenham(source, robot))
    wall_hits = sum(world[y, x] for x, y in centre_ray)
    delay_ns = range_m / LIGHT_SPEED * 1e9

    singular_values = np.linalg.svd(h, compute_uv=False)
    rank = np.linalg.matrix_rank(h, tol=1e-12)
    condition = singular_values[0] / max(singular_values[-1], 1e-30)
    total_rx_power_w = TX_POWER_W / N_TX * np.sum(np.abs(h) ** 2)
    snr_linear = total_rx_power_w / NOISE_POWER_W
    capacity_bps_hz = np.sum(
        np.log2(1 + (TX_POWER_W / (N_TX * NOISE_POWER_W)) * singular_values**2)
    )
    per_rx_dbm = db(TX_POWER_W / N_TX * np.sum(np.abs(h) ** 2, axis=1)) + 30
    per_tx_gain_db = db(np.sum(np.abs(h) ** 2, axis=0))
    rx_gram = h @ h.conj().T
    tx_gram = h.conj().T @ h
    rx_correlation = np.abs(rx_gram[0, 1]) / np.sqrt(rx_gram[0, 0] * rx_gram[1, 1])
    tx_correlation = np.abs(tx_gram[0, 1]) / np.sqrt(tx_gram[0, 0] * tx_gram[1, 1])

    np.set_printoptions(precision=3, suppress=True)
    print("\n===== MIMO channel report (agent start position) =====")
    print(f"Array / carrier          : {N_TX} Tx × {N_RX} Rx ULA, {CARRIER_HZ / 1e9:.3f} GHz")
    print(f"Wavelength / spacing     : {WAVELENGTH:.4f} m / {ELEMENT_SPACING:.4f} m (λ/2)")
    print(f"Tx total power / noise   : {TX_POWER_W * 1e3:.1f} mW (20 dBm) / {db(NOISE_POWER_W) + 30:.1f} dBm")
    print(f"Range / direct delay     : {range_m:.2f} m / {delay_ns:.2f} ns")
    print(f"AoD (source→robot)       : {aod_deg:.2f}°")
    print(f"AoA (robot→source)       : {aoa_deg:.2f}°")
    print(f"Wall cells on direct ray : {wall_hits}")
    print(f"H shape / rank           : {h.shape} / {rank}")
    print(f"|H| (linear amplitude)   :\n{np.abs(h)}")
    print(f"∠H (degrees)             :\n{np.degrees(np.angle(h))}")
    print(f"Singular values           : {singular_values}")
    print(f"Condition number          : {condition:.3e}")
    print(f"Frobenius norm²           : {np.linalg.norm(h, 'fro') ** 2:.3e}")
    print(f"Total Rx power            : {total_rx_power_w * 1e3:.3e} mW ({db(total_rx_power_w) + 30:.2f} dBm)")
    print(f"Per-Rx power              : {per_rx_dbm} dBm")
    print(f"Per-Tx channel gain       : {per_tx_gain_db} dB")
    print(f"Post-channel SNR          : {db(snr_linear):.2f} dB")
    print(f"Shannon capacity          : {capacity_bps_hz:.3f} bit/s/Hz")
    print(f"Rx / Tx adjacent corr.   : {rx_correlation:.4f} / {tx_correlation:.4f}")


def main():
    world = make_world()
    path = astar(AGENT, SOURCE, world)
    power = received_power_map(world, SOURCE)
    h_start = mimo_channel(AGENT, SOURCE, world)
    start_power_db, start_rank = mimo_metrics(h_start)
    tx_pilot, rx_signal = transmit_mimo_pilot(h_start, np.random.default_rng(7))

    fig, ax = plt.subplots(figsize=(10, 7))
    field = ax.imshow(power, origin="upper", extent=(0, NX, NY, 0), cmap="plasma")
    # Mask all free cells, leaving only the walls as solid black geometry.
    walls_only = np.ma.masked_where(~world, world)
    ax.imshow(walls_only, origin="upper", extent=(0, NX, NY, 0),
              cmap=ListedColormap(["black"]), interpolation="nearest")

    px, py = zip(*path)
    ax.plot(np.array(px) + 0.5, np.array(py) + 0.5, color="white", lw=2.5,
            label="A* reachable route")
    ax.scatter(AGENT[0] + 0.5, AGENT[1] + 0.5, s=100, c="deepskyblue",
               edgecolors="black", zorder=5, label="Agent")
    ax.scatter(SOURCE[0] + 0.5, SOURCE[1] + 0.5, s=140, c="crimson", marker="*",
               edgecolors="black", zorder=5, label="RF source")

    ax.set_title("2D RF source-seeking: 4×4 MIMO received-power map")
    ax.set_xlabel("x grid cell")
    ax.set_ylabel("y grid cell")
    ax.set_xlim(0, NX)
    ax.set_ylim(NY, 0)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    fig.colorbar(field, ax=ax, label="Mean MIMO channel power (dB, unit Tx power)")
    plt.tight_layout()
    print(f"Shortest reachable path: {len(path) - 1} grid steps")
    print(f"MIMO array: {N_TX} Tx x {N_RX} Rx, {CARRIER_HZ / 1e9:.1f} GHz")
    print(f"At start: mean channel power = {start_power_db:.1f} dB, rank = {start_rank}")
    print(f"Pilot simulation: Tx shape {tx_pilot.shape}, Rx signal shape {rx_signal.shape}")
    channel_report(h_start, AGENT, SOURCE, world)
    plt.show()


if __name__ == "__main__":
    main()
