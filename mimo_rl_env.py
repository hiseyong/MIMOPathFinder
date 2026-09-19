"""RF-only navigation environment built on RF_source_seeking_2D.py.

The policy never receives the occupancy map, robot coordinates, or source
coordinates.  They are used internally only for propagation and reward.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import pi

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from RF_source_seeking_2D import (AGENT, N_RX, N_TX, NX, NY, SOURCE, bfs_distance_field,
                                  generate_random_world, make_world, mimo_channel)


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 160
    history_length: int = 4
    source_radius: float = 1.5


# Fixed regardless of any env instance's own `seed`, so every env -- the one
# collecting rollouts and the ones evaluate() builds separately -- sees the
# identical map pool and the same train/holdout partition of maps.
MAP_POOL_SEED = 20240519
N_TRAIN_MAPS = 5
N_HOLDOUT_MAPS = 3

_MAP_POOL: list[dict] | None = None


def _build_map_pool() -> list[dict]:
    """Canonical map plus randomly generated ones, split train/holdout by map.

    Index 0 is always the original fixed room (make_world/AGENT/SOURCE), kept
    as one of the training maps for continuity with earlier runs and the
    fixed-start demo in evaluate_policy.py. The rest are generated once (see
    RF_source_seeking_2D.generate_random_world) with varied wall layout and
    source position, all satisfying the same "moderate difficulty" bound
    (<=2 walls on the direct line to source, <=3 corners on the shortest
    route). Holdout maps are never selected by reset(split="train"), so
    success there measures generalisation across environments, not just
    across start cells within one memorised map.
    """
    world0 = make_world()
    start_band0 = [(x, y) for y in range(20, NY - 1) for x in range(1, NX - 1) if not world0[y, x]]
    pool = [{"world": world0, "source": SOURCE, "start_band": start_band0,
            "distance_field": bfs_distance_field(world0, SOURCE)}]

    pool_rng = np.random.default_rng(MAP_POOL_SEED)
    for _ in range(N_TRAIN_MAPS - 1 + N_HOLDOUT_MAPS):
        world, source, start_band = generate_random_world(pool_rng)
        pool.append({"world": world, "source": source, "start_band": start_band,
                     "distance_field": bfs_distance_field(world, source)})
    return pool


def _get_map_pool() -> list[dict]:
    global _MAP_POOL
    if _MAP_POOL is None:
        _MAP_POOL = _build_map_pool()
    return _MAP_POOL


class MIMORFNavigationEnv:
    """POMDP: RF history in, relative motion action out."""
    FORWARD, LEFT, RIGHT, BACK = range(4)
    ACTION_NAMES = ("forward", "left + forward", "right + forward", "turn around + forward")
    DIRECTIONS = np.array(((1, 0), (0, 1), (-1, 0), (0, -1)), dtype=int)

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.map_pool = _get_map_pool()
        self._select_map(0)
        self.position, self.heading = AGENT, 0
        self.steps, self.last_action, self._remaining_distance = 0, self.FORWARD, 0
        self.rf_history: deque[np.ndarray] = deque(maxlen=self.config.history_length)

    def _select_map(self, index: int) -> None:
        # A map's distance field is precomputed once in the pool (map/source
        # are fixed per index), not re-run on every reset.
        m = self.map_pool[index]
        self.map_index = index
        self.world, self.source, self.start_band, self.distance_field = (
            m["world"], m["source"], m["start_band"], m["distance_field"]
        )

    def _sample_map_index(self, split: str) -> int:
        if split == "train":
            return int(self.rng.integers(N_TRAIN_MAPS))
        if split == "holdout":
            return N_TRAIN_MAPS + int(self.rng.integers(len(self.map_pool) - N_TRAIN_MAPS))
        return int(self.rng.integers(len(self.map_pool)))

    @property
    def observation_size(self) -> int:
        return self.config.history_length * (2 * N_RX * N_TX + N_RX + 1 + self.action_size)

    @property
    def action_size(self) -> int:
        return len(self.ACTION_NAMES)

    def _sample_start(self) -> tuple[int, int]:
        return self.start_band[int(self.rng.integers(len(self.start_band)))]

    def _feature(self) -> np.ndarray:
        # rx_heading is the robot's own orientation, never the hidden source bearing.
        h = mimo_channel(self.position, self.source, self.world, rx_heading=self.heading * pi / 2)
        amp = np.log10(np.abs(h) + 1e-15).ravel()
        phase = (np.angle(h) / pi).ravel()
        singular = np.log10(np.linalg.svd(h, compute_uv=False) + 1e-15)
        mean_power = np.array([np.log10(np.mean(np.abs(h) ** 2) + 1e-30)])
        action = np.eye(self.action_size, dtype=np.float32)[self.last_action]
        return np.concatenate((amp, phase, singular, mean_power, action)).astype(np.float32)

    def _observation(self) -> np.ndarray:
        return np.concatenate(tuple(self.rf_history), dtype=np.float32)

    def _oracle_path_distance(self, point: tuple[int, int]) -> int:
        # Privileged for training reward only; never returned in an observation.
        # Wall-aware BFS graph distance, never Euclidean/straight-line distance.
        return int(self.distance_field[point[1], point[0]])

    def _optimal_headings(self, point: tuple[int, int]) -> list[int]:
        """All headings that step onto an actually reachable shortest route.

        Every heading whose neighbour cell's BFS distance is exactly one less
        than the current cell's is an equally correct "answer" direction: by
        construction these are never wall crossings, and ties (several
        equally short routes around an obstacle) are all treated as correct
        instead of arbitrarily picking one path like a single A* run would.
        """
        here = self.distance_field[point[1], point[0]]
        headings = []
        for heading, (dx, dy) in enumerate(self.DIRECTIONS):
            nx, ny = point[0] + dx, point[1] + dy
            if (0 <= nx < NX and 0 <= ny < NY and not self.world[ny, nx]
                    and self.distance_field[ny, nx] == here - 1):
                headings.append(heading)
        return headings

    def reset(self, *, seed: int | None = None, random_start: bool = True,
              split: str = "train") -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if random_start:
            self._select_map(self._sample_map_index(split))
            self.position, self.heading = self._sample_start(), int(self.rng.integers(4))
        else:
            self._select_map(0)
            self.position, self.heading = AGENT, 0
        self.steps, self.last_action = 0, self.FORWARD
        self._remaining_distance = self._oracle_path_distance(self.position)
        self.rf_history.clear()
        feature = self._feature()
        for _ in range(self.config.history_length):
            self.rf_history.append(feature.copy())
        return self._observation()

    def step(self, action: int):
        if action not in (self.FORWARD, self.LEFT, self.RIGHT, self.BACK):
            raise ValueError(f"Unknown action: {action}")
        old_distance = self._remaining_distance
        optimal_headings = self._optimal_headings(self.position)
        if action == self.LEFT:
            self.heading = (self.heading - 1) % 4
        elif action == self.RIGHT:
            self.heading = (self.heading + 1) % 4
        elif action == self.BACK:
            self.heading = (self.heading + 2) % 4
        # Reward the direction actually chosen this step against every
        # wall-aware shortest-route heading, not the straight line to source.
        on_shortest_route = self.heading in optimal_headings
        candidate = tuple(self.position + self.DIRECTIONS[self.heading])
        collision = self.world[candidate[1], candidate[0]]
        if not collision:
            self.position = candidate
            self._remaining_distance = self._oracle_path_distance(self.position)
        self.steps += 1
        reached = np.linalg.norm(np.asarray(self.position) - np.asarray(self.source)) <= self.config.source_radius
        timeout, terminated = self.steps >= self.config.max_steps, reached or self.steps >= self.config.max_steps
        reward = 0.35 * (old_distance - self._remaining_distance) - 0.02
        if optimal_headings:
            reward += 0.3 if on_shortest_route else -0.1
        reward += -0.8 if collision else 0.0
        reward += 20.0 if reached else (-2.0 if timeout else 0.0)
        self.last_action = action
        self.rf_history.append(self._feature())
        return self._observation(), float(reward), terminated, {"reached": reached, "collision": collision, "steps": self.steps}

    def expert_action(self) -> int:
        """Wall-aware teacher action, available only while constructing training data."""
        optimal_headings = self._optimal_headings(self.position)
        if not optimal_headings:
            return self.FORWARD
        turn = (optimal_headings[0] - self.heading) % 4
        return {0: self.FORWARD, 1: self.RIGHT, 2: self.BACK, 3: self.LEFT}[turn]

    def render(self, trajectory: list[tuple[int, int]] | None = None):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(self.world, origin="upper", cmap="gray_r", interpolation="nearest")
        if trajectory:
            x, y = zip(*trajectory)
            ax.plot(x, y, color="deepskyblue", lw=2, label="policy trajectory")
        ax.scatter(*self.source, marker="*", s=190, c="crimson", label="RF source", zorder=3)
        ax.scatter(*self.position, s=70, c="lime", edgecolors="black", label="agent", zorder=3)
        ax.set(xlim=(0, NX - 1), ylim=(NY - 1, 0), aspect="equal", title="RF-only MIMO navigation")
        ax.legend(loc="lower right")
        return fig

    def render_animation(self, trajectory: list[tuple[int, int]], interval: int = 150) -> FuncAnimation:
        """Animate the agent moving step by step, revealing the route as taken."""
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(self.world, origin="upper", cmap="gray_r", interpolation="nearest")
        ax.scatter(*self.source, marker="*", s=190, c="crimson", label="RF source", zorder=3)
        trail, = ax.plot([], [], color="deepskyblue", lw=2, label="policy trajectory")
        agent = ax.scatter([], [], s=90, c="lime", edgecolors="black", label="agent", zorder=4)
        step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        ax.set(xlim=(0, NX - 1), ylim=(NY - 1, 0), aspect="equal", title="RF-only MIMO navigation")
        ax.legend(loc="lower right")

        def update(frame):
            xs, ys = zip(*trajectory[:frame + 1])
            trail.set_data(xs, ys)
            agent.set_offsets([trajectory[frame]])
            step_text.set_text(f"step {frame}/{len(trajectory) - 1}")
            return trail, agent, step_text

        return FuncAnimation(fig, update, frames=len(trajectory), interval=interval, repeat=False)
