# MIMOPathFinder

RF-only source-seeking: an agent must reach a hidden RF source in an unknown
indoor environment using only a history of MIMO channel-state information
(CSI) — never the occupancy map, its own coordinates, or the source
location. A GRU policy outputs a continuous heading direction; training is
DAgger against a privileged, wall-aware BFS shortest-path teacher (see
`simple_2D/abstract.txt` for the full research writeup and measured numbers).

## Project layout

- **`simple_2D/`** — archived first-phase codebase. A procedurally generated
  2D grid world (one fixed room plus randomly generated layouts, bounded to
  moderate visibility complexity) used to develop the reward design and
  training methodology. Frozen; final result was 83.8% success on held-out
  procedurally generated maps. See `simple_2D/README.md`.

- **Top level (this directory)** — current codebase, built on the
  **ADWA IROS-2020 benchmark**: 17 real building floorplans (ROS
  map_server-style occupancy grids + recorded navigation tasks), extracted
  under `adwa_benchmark/`. This replaces the toy procedural maps with real
  floorplan statistics for a strictly harder generalisation test.

  - `rf_physics.py` — grid-size-agnostic MIMO channel model, BFS/A*
    pathfinding (generalised from `simple_2D/RF_source_seeking_2D.py`, which
    hardcoded one grid size).
  - `adwa_maps.py` — loads a building's PNG occupancy map + yaml + recorded
    tasks, downsampled to a configurable cell size; keeps only the largest
    connected free-cell component (conservative downsampling can close a
    doorway narrower than one cell).
  - `adwa_env.py` — `ADWANavigationEnv`: same POMDP/reward design as
    `simple_2D/mimo_rl_env.py`, but the map pool is real buildings split
    train/holdout by building (`adwa_env.HOLDOUT_BUILDINGS`).
  - `train_adwa_dagger.py` — same iterative DAgger loop as
    `simple_2D/train_dagger.py`, pointed at `adwa_env`.
  - `evaluate_adwa_policy.py` — renders one policy rollout (PNG + step-by-step
    GIF) on a holdout building by default.

```bash
poetry lock
poetry install
poetry run python train_adwa_dagger.py --rounds 10 --episodes-per-round 150
poetry run python evaluate_adwa_policy.py
```

Training saves `checkpoints/adwa_gru_dagger.pt` (best holdout-building
success) and a resumable `checkpoints/adwa_dagger_resume.pt` (pass `--resume`
to continue an interrupted run instead of restarting).

## Model architecture

### Observation (input)

Every environment step, `adwa_env._feature()` (`rf_physics.mimo_channel`
under the hood) turns the current 4x4 narrowband MIMO channel matrix `H`
into a 39-dim real feature vector:

| Component | Source | Dims |
|---|---|---|
| `amp` | `log10(|H|)`, flattened | `N_RX*N_TX` = 16 |
| `phase` | `angle(H)/pi`, flattened | `N_RX*N_TX` = 16 |
| `singular` | `log10(svd(H))` | `N_RX` = 4 |
| `mean_power` | `log10(mean(|H|^2))` | 1 |
| `last_direction` | unit vector of the previous executed heading | 2 |

The policy never sees `H` itself, the occupancy map, its own coordinates, or
the source location — only this derived, privacy-preserving feature. The
environment keeps a sliding window of the last `history_length=4` such
frames and concatenates them into the actual network input:

```
observation_size = history_length * 39 = 4 * 39 = 156
```

### Network (`train_adwa_dagger.RecurrentPolicy`, 252,673 parameters)

```
observation (156)
     │
     ▼
Linear(156 → 192) + Tanh          "encoder"            30,144 params
     │
     ▼
GRU(192 → 192), 1 layer           "gru"               222,336 params
     │  (hidden state persists across the WHOLE episode,
     │   reset only at episode boundaries -- so the policy's
     │   effective memory is longer than the explicit 4-frame
     │   window above)
     ▼
Linear(192 → 1)                   "actor"                 193 params
     │
     ▼
   theta  (unconstrained scalar angle, radians)
     │
     ▼
(cos(theta), sin(theta))          -> unit heading direction
```

The actor head regresses a single unbounded angle rather than a 2D vector or
a 4-way action logit, because `(cos θ, sin θ)` is *always* exactly unit norm
by construction. This sidesteps two failure modes a naive continuous-control
head would have: normalising a raw `Linear(192 → 2)` output is unstable near
the zero vector (undefined direction, exploding gradient), and regressing
`theta` directly with an MSE-style loss is discontinuous across the `-π/π`
wraparound (e.g. `-179°` and `+179°` are 2° apart physically but ~358° apart
in raw angle space). Feeding `theta` through `cos`/`sin` and comparing the
resulting unit vectors makes the loss wrap-safe automatically.

`greedy_direction()` (evaluation/deployment) is fully deterministic — no
sampling distribution, no exploration noise — since the network's output
*is* the action.

### Training objective (DAgger, not policy-gradient RL)

Every training step, `adwa_env.expert_direction()` returns the wall-aware
shortest-path direction from a BFS potential field seeded at the source
(privileged: uses the true map, never available to the deployed policy).
`train_adwa_dagger.collect_round()` executes a decaying mixture of this
teacher direction and the policy's own predicted direction, but always
labels the visited state with the teacher's direction. The network is
trained with:

```
loss = mean( 1 - cos_similarity(predicted_direction, teacher_direction) )
```

i.e. pure direction regression, no reward, no value function, no
policy-gradient/advantage estimation — see `simple_2D/abstract.txt` for why this
problem (a privileged optimal-direction oracle available at every training
step) is a better fit for imitation learning than for PPO-style RL, and the
measured evidence (PPO collapsing a pretrained policy to 0% success) that
motivated the switch.

### Reward (diagnostic only — not part of the training loss)

`adwa_env.step()` still computes a reward for `evaluate()`'s `reward_sum`
diagnostic and for anyone who wants to plug in an RL algorithm later:

```
reward = 0.35 * (BFS_distance_before - BFS_distance_after)   # progress
       - 0.02                                                # step cost
       + 0.3 * cos_similarity(action, best_teacher_direction) # alignment
       - 0.8  if collision else 0
       + 20.0 if reached else (-2.0 if timeout else 0)
```

Both the distance term and the alignment term are computed on a BFS
shortest-path field over the free-cell graph, never Euclidean distance to
the source — so a policy is never rewarded for heading in the straight-line
direction of a source it cannot actually reach that way.
