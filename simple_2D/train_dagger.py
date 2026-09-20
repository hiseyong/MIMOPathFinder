"""Train a GRU policy via DAgger on RF-only navigation.

The policy outputs a continuous 2D heading direction (not one of four
discrete grid actions): the network regresses a single angle theta, and
cos/sin(theta) is used as the unit movement direction, so it is free to move
at any heading rather than only forward/left/right/back turns snapped to the
grid. Each round rolls out the current policy (mixed with the wall-aware BFS
teacher, decaying towards pure policy rollout), labels every visited state
with the teacher *direction* regardless of which direction was actually
executed, aggregates the dataset across rounds, and retrains by minimising
1 - cosine_similarity(prediction, teacher) -- direction-only regression, scale
invariant and free of the angle-wraparound discontinuity a raw theta MSE loss
would have.

Why DAgger instead of PPO fine-tuning: the environment hands a privileged,
always-available, wall-aware optimal direction
(mimo_rl_env.MIMORFNavigationEnv.expert_direction, via a BFS distance field)
at every training step, so this is fundamentally a supervised imitation
problem, not a sparse-reward RL problem. Measured on this codebase in the
discrete-action version: pretrain-only success scaled cleanly with data (5%
at 400 episodes -> 17.5% at 1200), while PPO fine-tuning collapsed that same
pretrained policy to 0% success almost immediately and never recovered.

Example: poetry run python train_dagger.py --rounds 10 --episodes-per-round 150
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mimo_rl_env import EnvConfig, MIMORFNavigationEnv


class RecurrentPolicy(nn.Module):
    """GRU encoder with a single-angle actor head: cos/sin(theta) is always a
    unit vector by construction, so this avoids both the degenerate-zero-
    vector instability of normalising a raw 2D output and the discontinuity
    of regressing theta directly across the -pi/pi wrap.
    """

    def __init__(self, observation_size: int, hidden_size: int = 192):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh())
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.actor, self.hidden_size = nn.Linear(hidden_size, 1), hidden_size

    def forward_sequence(self, observations, episode_starts, initial_hidden=None):
        hidden = (torch.zeros(1, 1, self.hidden_size, device=observations.device)
                  if initial_hidden is None else initial_hidden.clone())
        thetas = []
        for t in range(observations.shape[0]):
            hidden = hidden * (1.0 - episode_starts[t].float().view(1, 1, 1))
            output, hidden = self.gru(self.encoder(observations[t]).view(1, 1, -1), hidden)
            thetas.append(self.actor(output[0, 0]).squeeze(-1))
        return torch.stack(thetas)

    @torch.no_grad()
    def greedy_direction(self, observation, hidden, episode_start):
        if episode_start:
            hidden.zero_()
        output, hidden = self.gru(self.encoder(observation).view(1, 1, -1), hidden)
        theta = self.actor(output[0, 0]).squeeze(-1)
        direction = torch.stack([torch.cos(theta), torch.sin(theta)])
        return direction.cpu().numpy(), hidden


@torch.no_grad()
def evaluate(policy: RecurrentPolicy, env: MIMORFNavigationEnv, device: torch.device,
             episodes: int = 60, split: str = "train") -> tuple[float, float]:
    """Deterministic (greedy) success rate and mean episode reward.

    split="holdout" starts only from maps never drawn during split="train"
    rollouts, so it measures generalisation to unseen environments rather
    than just an unlucky random draw from the same trained-on distribution.
    """
    policy.eval()
    successes, reward_sum = 0, 0.0
    for _ in range(episodes):
        observation, done, episode_start = env.reset(split=split), False, True
        hidden = torch.zeros(1, 1, policy.hidden_size, device=device)
        while not done:
            direction, hidden = policy.greedy_direction(
                torch.as_tensor(observation, device=device), hidden, episode_start
            )
            observation, reward, done, info = env.step(direction)
            reward_sum += reward
            episode_start = done
        successes += int(info["reached"])
    policy.train()
    return successes / episodes, reward_sum / episodes


def collect_round(policy: RecurrentPolicy, env: MIMORFNavigationEnv, episodes: int,
                   device: torch.device, beta: float):
    """Roll out a per-step beta-mix of teacher/policy directions.

    Every visited state is labelled with the teacher direction regardless of
    which one was actually executed -- the state distribution comes
    increasingly from the learner's own (possibly imperfect) choices as beta
    decays across rounds, which is what lets DAgger correct states a fixed
    demonstration set would never cover.
    """
    demonstrations = []
    successes, reward_sum = 0, 0.0
    for _ in range(episodes):
        observation, done = env.reset(), False
        hidden = torch.zeros(1, 1, policy.hidden_size, device=device)
        observations, targets = [], []
        while not done:
            teacher_direction = env.expert_direction()
            if env.rng.random() < beta:
                executed = teacher_direction
            else:
                with torch.no_grad():
                    output, hidden = policy.gru(
                        policy.encoder(torch.as_tensor(observation, device=device)).view(1, 1, -1), hidden
                    )
                    theta = policy.actor(output[0, 0]).squeeze(-1)
                    executed = torch.stack([torch.cos(theta), torch.sin(theta)]).cpu().numpy()
            observations.append(observation)
            targets.append(teacher_direction)
            observation, reward, done, info = env.step(executed)
            reward_sum += reward
        demonstrations.append((np.asarray(observations, dtype=np.float32),
                               np.asarray(targets, dtype=np.float32)))
        successes += int(info["reached"])
    return demonstrations, successes / episodes, reward_sum / episodes


def trim_replay(demonstrations: list, cap: int) -> list:
    """Evict oldest episodes (FIFO) once the aggregated dataset exceeds cap.

    Unbounded aggregation both keeps making each round slower (contributing
    to this codebase's repeated background-kill/timeout problem at ~300k
    samples) and dilutes the most useful data: the states the *current*
    policy actually visits under a decayed beta. A capped, recency-biased
    buffer keeps round time roughly constant and keeps training focused on
    the states that currently matter most, at some cost to early, more
    teacher-heavy coverage.
    """
    total = sum(len(t) for _, t in demonstrations)
    while total > cap and len(demonstrations) > 1:
        total -= len(demonstrations.pop(0)[1])
    return demonstrations


def train_on_dataset(policy, optimizer, demonstrations, device, epochs, group_size):
    for _ in range(epochs):
        order = np.random.permutation(len(demonstrations))
        for indices in np.array_split(order, max(1, len(order) // group_size)):
            sequences = [demonstrations[i] for i in indices]
            lengths = [len(targets) for _, targets in sequences]
            obs = torch.as_tensor(np.concatenate([features for features, _ in sequences]), device=device)
            labels = torch.as_tensor(np.concatenate([targets for _, targets in sequences]), device=device)
            starts = torch.zeros(len(labels), device=device)
            starts[np.cumsum([0, *lengths[:-1]])] = 1.0
            thetas = policy.forward_sequence(obs, starts)
            predicted = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=-1)
            loss = (1.0 - (predicted * labels).sum(-1)).mean()  # 1 - cosine similarity
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 0.5); optimizer.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--episodes-per-round", type=int, default=150)
    parser.add_argument("--epochs-per-round", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=8, help="Episodes per gradient step.")
    parser.add_argument("--beta0", type=float, default=1.0, help="Round-1 probability of executing the teacher direction.")
    parser.add_argument("--beta-decay", type=float, default=0.8, help="Multiplicative beta decay per round.")
    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument("--eval-episodes", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-decay", type=float, default=0.97, help="Multiplicative LR decay per round.")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="L2 regularisation, to discourage memorising the training-map set.")
    parser.add_argument("--replay-cap", type=int, default=80_000,
                        help="Max aggregated transitions kept; oldest episodes evicted first.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from checkpoints/dagger_resume.pt if present, instead of round 1.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MIMORFNavigationEnv(EnvConfig(), args.seed)
    policy = RecurrentPolicy(env.observation_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Path("checkpoints").mkdir(exist_ok=True)
    resume_path = Path("checkpoints/dagger_resume.pt")

    start_round, all_demonstrations, best_holdout_success, beta = 1, [], -1.0, args.beta0
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        policy.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        all_demonstrations = state["demonstrations"]
        best_holdout_success = state["best_holdout_success"]
        beta = state["beta"]
        start_round = state["round_idx"] + 1
        print(f"Resumed from round {state['round_idx']}: beta={beta:.3f} "
              f"dataset={sum(len(t) for _, t in all_demonstrations)} best-holdout={best_holdout_success:.1%}")

    for round_idx in range(start_round, args.rounds + 1):
        demos, rollout_success, rollout_reward = collect_round(policy, env, args.episodes_per_round, device, beta)
        all_demonstrations.extend(demos)
        all_demonstrations = trim_replay(all_demonstrations, args.replay_cap)
        loss = train_on_dataset(policy, optimizer, all_demonstrations, device, args.epochs_per_round, args.group_size)
        eval_env = MIMORFNavigationEnv(seed=args.seed + round_idx)
        train_success, train_reward = evaluate(policy, eval_env, device, args.eval_episodes, split="train")
        holdout_success, holdout_reward = evaluate(policy, eval_env, device, args.eval_episodes, split="holdout")
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"round={round_idx:>2}/{args.rounds} beta={beta:.2f} lr={lr_now:.2e} "
              f"dataset={sum(len(t) for _, t in all_demonstrations):>7} "
              f"rollout-success={rollout_success:.1%} rollout-reward={rollout_reward:6.2f} "
              f"train-eval={train_success:.1%} ({train_reward:6.2f}) "
              f"holdout-eval={holdout_success:.1%} ({holdout_reward:6.2f}) loss={loss:.3f}")
        if holdout_success >= best_holdout_success:
            best_holdout_success = holdout_success
            torch.save({"model": policy.state_dict(), "observation_size": env.observation_size,
                       "env_config": asdict(env.config)}, "checkpoints/rf_gru_dagger.pt")
        beta = max(args.beta_min, beta * args.beta_decay)
        for group in optimizer.param_groups:
            group["lr"] *= args.lr_decay
        # Saved every round (not just on improvement) so a killed/interrupted
        # run can resume from the last completed round via --resume, instead
        # of losing all rollout/training compute done so far.
        torch.save({"round_idx": round_idx, "beta": beta, "model": policy.state_dict(),
                   "optimizer": optimizer.state_dict(), "demonstrations": all_demonstrations,
                   "best_holdout_success": best_holdout_success}, resume_path)

    print(f"Best holdout-map success={best_holdout_success:.1%}; saved checkpoints/rf_gru_dagger.pt")


if __name__ == "__main__":
    main()
