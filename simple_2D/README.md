# simple_2D (archived)

Frozen first-phase codebase: a procedurally generated 2D grid world (a fixed
room plus randomly generated wall-connected layouts satisfying a bounded
visibility complexity), used to develop and validate the reward design and
training methodology before moving to the real building floorplans in
`../adwa_benchmark` (see the top-level README). Run everything from inside
this directory (`cd simple_2D`); its scripts import each other by module
name and are not meant to be mixed with the top-level ADWA-based codebase.

**Final result**: continuous-heading DAgger policy over a 33-map pool (25
train + 8 entirely held-out maps/source positions) reached **83.8% success
on unseen holdout maps** (`checkpoints/rf_gru_dagger.pt`). See `abstract.txt`
for the full writeup (English + Korean) of the reward redesign, the PPO
methodology evaluation and its collapse, the discrete-to-continuous action
migration, and the overfitting diagnosis that motivated the multi-map pool.

RF-only source-seeking in an unknown indoor environment. The policy gets a
history of MIMO CSI amplitude/phase, singular values, mean channel power,
and its prior action—never map cells, robot coordinates, source
coordinates, or collision state.

```bash
cd simple_2D
poetry run python train_dagger.py --rounds 10 --episodes-per-round 150
poetry run python evaluate_policy.py
```

Training saves `checkpoints/rf_gru_dagger.pt`; evaluation saves
`outputs/evaluation_trajectory.png`.

The simulator uses the hidden map and source position only to propagate the
RF channel, compute a wall-aware BFS distance-to-source field, and derive the
teacher action. Neither is ever part of the observation given to the GRU
policy.

The action space is a continuous heading direction (not the four discrete
grid actions of an earlier iteration): the network regresses a single angle
`theta`, and `(cos theta, sin theta)` is the unit direction the agent moves
along each step, so it is never snapped to a grid axis. Training is DAgger:
each round rolls out a decaying mix of the teacher and the current policy,
labels every visited state with the wall-aware teacher *direction*
regardless of which one was executed, and retrains by minimising
`1 - cosine_similarity(prediction, teacher)` on the aggregated dataset — the
teacher/BFS field is never available to the deployed policy. Console
`holdout-eval=` is a deterministic held-out success rate; use it rather than
`rollout-success=` (collected under the exploring, non-greedy mixed policy)
to judge whether the policy is improving. The network itself (GRU encoder +
single-angle actor head) is architecturally identical to the one used in the
ADWA-based codebase — see the "Model architecture" section of the top-level
`../README.md` for the full breakdown (layer sizes, parameter counts, and
why the actor head regresses an angle rather than a raw 2D vector).

`train_recurrent_ppo.py` (PPO fine-tuning on top of the same imitation
pretrain) is kept for reference but deprecated: measured on this codebase, it
collapses the pretrained policy's success rate to 0% almost immediately and
never recovers, even after fixing a real GRU hidden-state bug in its
recomputed log-probs. Pure imitation, by contrast, scaled cleanly with more
data (5% success at 400 episodes -> 17.5% at 1200) — see the module
docstring in `train_recurrent_ppo.py` for the measured numbers.
