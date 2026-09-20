"""RF-only navigation environment over real ADWA-benchmark building floorplans.

Successor to simple_2D/mimo_rl_env.py: same POMDP design (RF-CSI history in,
continuous heading direction out; map/coordinates/source never observed) and
same wall-aware BFS-potential reward, but the map pool is now real building
floorplans (see adwa_maps.py) split train/holdout BY BUILDING, instead of
procedurally generated toy rooms. This is a strictly harder and more
realistic generalisation test: different buildings have genuinely different
floorplan statistics, not just different instances of the same generator.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import pi

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from adwa_maps import list_buildings, load_map
from rf_physics import N_RX, N_TX, bfs_distance_field, mimo_channel, neighbours

# Fixed regardless of any env instance's own `seed`, so every env -- the one
# collecting rollouts and the ones evaluate() builds separately -- sees the
# identical building split.
HOLDOUT_BUILDINGS = ("Eastville", "Mosquito", "Sisters2", "Scioto2")


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 300
    history_length: int = 4
    source_radius_m: float = 0.6
    step_size_m: float = 0.3
    cell_size_m: float = 0.3
    min_start_source_distance_m: float = 3.0


_MAP_POOL: dict[str, dict] | None = None


def _largest_connected_component(world: np.ndarray) -> list[tuple[int, int]]:
    """Free cells reachable from each other, discarding smaller disconnected
    pockets -- downsampling a real floorplan by conservative max-pooling can
    close a doorway narrower than one cell, splitting off an unreachable
    region; sampling start/source only from the largest component guarantees
    every episode is actually solvable.
    """
    all_free = [(x, y) for y in range(world.shape[0])
                for x in range(world.shape[1]) if not world[y, x]]
    unvisited, best = set(all_free), []
    while unvisited:
        seed = next(iter(unvisited))
        component, frontier = [], deque([seed])
        unvisited.discard(seed)
        while frontier:
            current = frontier.popleft()
            component.append(current)
            for nxt in neighbours(current, world):
                if nxt in unvisited:
                    unvisited.discard(nxt)
                    frontier.append(nxt)
        if len(component) > len(best):
            best = component
    return best


def _build_map_pool(cell_size_m: float) -> dict[str, dict]:
    pool = {}
    for name in list_buildings():
        building = load_map(name, cell_size_m=cell_size_m)
        free_cells = _largest_connected_component(building.world)
        pool[name] = {"world": building.world, "free_cells": free_cells}
    return pool


def _get_map_pool(cell_size_m: float) -> dict[str, dict]:
    global _MAP_POOL
    if _MAP_POOL is None:
        _MAP_POOL = _build_map_pool(cell_size_m)
    return _MAP_POOL


class ADWANavigationEnv:
    """POMDP: RF history in, a continuous 2D heading direction out.

    Same interface as simple_2D.mimo_rl_env.MIMORFNavigationEnv, so the
    DAgger training loop in train_adwa_dagger.py is a near-verbatim reuse of
    simple_2D/train_dagger.py, just pointed at this environment.
    """

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.map_pool = _get_map_pool(self.config.cell_size_m)
        all_buildings = sorted(self.map_pool)
        self.holdout_buildings = [b for b in all_buildings if b in HOLDOUT_BUILDINGS]
        self.train_buildings = [b for b in all_buildings if b not in HOLDOUT_BUILDINGS]

        self.building_name = self.train_buildings[0]
        self._select_building(self.building_name)
        self.position, self.heading = np.array(self.free_cells[0], dtype=float), 0.0
        self.source = self.free_cells[0]
        self.last_direction = np.array([1.0, 0.0], dtype=np.float32)
        self.steps, self._remaining_distance = 0, 0
        self.rf_history: deque[np.ndarray] = deque(maxlen=self.config.history_length)

    def _select_building(self, name: str) -> None:
        m = self.map_pool[name]
        self.building_name = name
        self.world, self.free_cells = m["world"], m["free_cells"]

    def _sample_building(self, split: str) -> str:
        pool = self.train_buildings if split == "train" else (
            self.holdout_buildings if split == "holdout" else sorted(self.map_pool))
        return pool[int(self.rng.integers(len(pool)))]

    @property
    def observation_size(self) -> int:
        return self.config.history_length * (2 * N_RX * N_TX + N_RX + 1 + 2)

    def _sample_position(self) -> tuple[int, int]:
        return self.free_cells[int(self.rng.integers(len(self.free_cells)))]

    def _sample_start_source_pair(self) -> tuple[tuple[int, int], tuple[int, int]]:
        min_cells = self.config.min_start_source_distance_m / self.config.cell_size_m
        for _ in range(200):
            source = self._sample_position()
            start = self._sample_position()
            if np.hypot(start[0] - source[0], start[1] - source[1]) >= min_cells:
                return start, source
        return start, source  # fall back to whatever the last draw was

    def _feature(self) -> np.ndarray:
        h = mimo_channel(self.position, self.source, self.world, rx_heading=self.heading)
        amp = np.log10(np.abs(h) + 1e-15).ravel()
        phase = (np.angle(h) / pi).ravel()
        singular = np.log10(np.linalg.svd(h, compute_uv=False) + 1e-15)
        mean_power = np.array([np.log10(np.mean(np.abs(h) ** 2) + 1e-30)])
        return np.concatenate((amp, phase, singular, mean_power, self.last_direction)).astype(np.float32)

    def _observation(self) -> np.ndarray:
        return np.concatenate(tuple(self.rf_history), dtype=np.float32)

    def _cell(self, point) -> tuple[int, int]:
        ny, nx = self.world.shape
        x = int(np.clip(round(float(point[0])), 0, nx - 1))
        y = int(np.clip(round(float(point[1])), 0, ny - 1))
        return x, y

    def _oracle_path_distance(self, point) -> int:
        x, y = self._cell(point)
        return int(self.distance_field[y, x])

    def _optimal_headings(self, point) -> list[int]:
        ny, nx = self.world.shape
        cx, cy = self._cell(point)
        here = self.distance_field[cy, cx]
        headings = []
        for heading, (dx, dy) in enumerate(self.DIRECTIONS):
            px, py = cx + dx, cy + dy
            if (0 <= px < nx and 0 <= py < ny and not self.world[py, px]
                    and self.distance_field[py, px] == here - 1):
                headings.append(heading)
        return headings

    def _blocked(self, start: np.ndarray, end: np.ndarray) -> bool:
        for t in np.linspace(0.0, 1.0, 6)[1:]:
            x, y = self._cell(start + t * (end - start))
            if self.world[y, x]:
                return True
        return False

    DIRECTIONS = np.array(((1, 0), (0, 1), (-1, 0), (0, -1)), dtype=int)

    def reset(self, *, seed: int | None = None, split: str = "train") -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._select_building(self._sample_building(split))
        start, source = self._sample_start_source_pair()
        self.source = source
        self.distance_field = bfs_distance_field(self.world, self.source)
        self.position = np.array(start, dtype=float)
        self.heading = float(self.rng.uniform(-pi, pi))
        self.steps = 0
        self.last_direction = np.array([np.cos(self.heading), np.sin(self.heading)], dtype=np.float32)
        self._remaining_distance = self._oracle_path_distance(self.position)
        self.rf_history.clear()
        feature = self._feature()
        for _ in range(self.config.history_length):
            self.rf_history.append(feature.copy())
        return self._observation()

    def step(self, direction: np.ndarray):
        """direction: any real 2-vector; renormalised to a unit heading."""
        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        unit = direction / norm if norm > 1e-8 else np.array([np.cos(self.heading), np.sin(self.heading)])

        old_distance = self._remaining_distance
        optimal_headings = self._optimal_headings(self.position)
        best_alignment = max((float(unit @ self.DIRECTIONS[h]) for h in optimal_headings), default=0.0)

        self.heading = float(np.arctan2(unit[1], unit[0]))
        step_cells = self.config.step_size_m / self.config.cell_size_m
        candidate = self.position + step_cells * unit
        collision = self._blocked(self.position, candidate)
        if not collision:
            self.position = candidate
            self._remaining_distance = self._oracle_path_distance(self.position)
        self.steps += 1
        source_radius_cells = self.config.source_radius_m / self.config.cell_size_m
        reached = np.linalg.norm(self.position - np.asarray(self.source, dtype=float)) <= source_radius_cells
        timeout, terminated = self.steps >= self.config.max_steps, reached or self.steps >= self.config.max_steps
        reward = 0.35 * (old_distance - self._remaining_distance) - 0.02
        if optimal_headings:
            reward += 0.3 * best_alignment
        reward += -0.8 if collision else 0.0
        reward += 20.0 if reached else (-2.0 if timeout else 0.0)
        self.last_direction = unit.astype(np.float32)
        self.rf_history.append(self._feature())
        return self._observation(), float(reward), terminated, {
            "reached": reached, "collision": collision, "steps": self.steps, "building": self.building_name,
        }

    def expert_direction(self) -> np.ndarray:
        """Wall-aware teacher direction (unit vector), available only while constructing training data."""
        optimal_headings = self._optimal_headings(self.position)
        if not optimal_headings:
            return np.array([np.cos(self.heading), np.sin(self.heading)], dtype=np.float32)
        return self.DIRECTIONS[optimal_headings[0]].astype(np.float32)

    def render(self, trajectory: list[tuple[float, float]] | None = None):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(self.world, origin="upper", cmap="gray_r", interpolation="nearest")
        if trajectory:
            x, y = zip(*trajectory)
            ax.plot(x, y, color="deepskyblue", lw=2, label="policy trajectory")
        ax.scatter(*self.source, marker="*", s=190, c="crimson", label="RF source", zorder=3)
        ax.scatter(*self.position, s=70, c="lime", edgecolors="black", label="agent", zorder=3)
        ny, nx = self.world.shape
        ax.set(xlim=(0, nx - 1), ylim=(ny - 1, 0), aspect="equal",
               title=f"RF-only navigation -- {self.building_name}")
        ax.legend(loc="lower right")
        return fig

    def render_animation(self, trajectory: list[tuple[float, float]], interval: int = 80) -> FuncAnimation:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(self.world, origin="upper", cmap="gray_r", interpolation="nearest")
        ax.scatter(*self.source, marker="*", s=190, c="crimson", label="RF source", zorder=3)
        trail, = ax.plot([], [], color="deepskyblue", lw=2, label="policy trajectory")
        agent = ax.scatter([], [], s=90, c="lime", edgecolors="black", label="agent", zorder=4)
        step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        ny, nx = self.world.shape
        ax.set(xlim=(0, nx - 1), ylim=(ny - 1, 0), aspect="equal",
               title=f"RF-only navigation -- {self.building_name}")
        ax.legend(loc="lower right")

        def update(frame):
            xs, ys = zip(*trajectory[:frame + 1])
            trail.set_data(xs, ys)
            agent.set_offsets([trajectory[frame]])
            step_text.set_text(f"step {frame}/{len(trajectory) - 1}")
            return trail, agent, step_text

        return FuncAnimation(fig, update, frames=len(trajectory), interval=interval, repeat=False)
