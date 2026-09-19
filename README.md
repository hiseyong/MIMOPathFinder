# MIMOPathFinder

RF-only source-seeking in an unknown U-shaped indoor environment. The policy
gets a history of MIMO CSI amplitude/phase, singular values, mean channel
power, and its prior action—never map cells, robot coordinates, source
coordinates, or collision state.

```bash
poetry lock
poetry install
poetry run python train_dagger.py --rounds 10 --episodes-per-round 150
poetry run python evaluate_policy.py
```

Training saves `checkpoints/rf_gru_dagger.pt`; evaluation saves
`outputs/evaluation_trajectory.png`.

The simulator uses the hidden map and source position only to propagate the
RF channel, compute a wall-aware BFS distance-to-source field, and derive the
teacher action. Neither is ever part of the observation given to the GRU
policy.

The action space includes forward, left/right turn with motion, and a 180°
turn with motion. This lets the robot leave a one-cell-wide corridor or a dead
end. Training is DAgger: each round rolls out a decaying mix of the teacher
and the current policy, labels every visited state with the wall-aware
teacher action regardless of which one was executed, and retrains on the
aggregated dataset — the teacher/BFS field is never available to the
deployed policy. Console `eval-success=` is a deterministic held-out success
rate; use it rather than `rollout-success=` (collected under the exploring,
non-greedy mixed policy) to judge whether the policy is improving.

`train_recurrent_ppo.py` (PPO fine-tuning on top of the same imitation
pretrain) is kept for reference but deprecated: measured on this codebase, it
collapses the pretrained policy's success rate to 0% almost immediately and
never recovers, even after fixing a real GRU hidden-state bug in its
recomputed log-probs. Pure imitation, by contrast, scaled cleanly with more
data (5% success at 400 episodes -> 17.5% at 1200) — see the module
docstring in `train_recurrent_ppo.py` for the measured numbers.
