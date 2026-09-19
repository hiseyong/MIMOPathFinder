"""Train a GRU policy via DAgger on RF-only navigation.

Each round rolls out the current policy (mixed with the wall-aware BFS/A*
teacher, decaying towards pure policy rollout), labels every visited state
with the teacher action regardless of which action was actually executed,
aggregates the dataset across rounds, and retrains by supervised
cross-entropy. This replaces the PPO fine-tuning stage in
train_recurrent_ppo.py.

Why: the environment hands a privileged, always-available, wall-aware
optimal action (mimo_rl_env.MIMORFNavigationEnv.expert_action, via a BFS
distance field) at every training step, so this is fundamentally a
supervised imitation problem, not a sparse-reward RL problem. Measured on
this codebase: pretrain-only success scaled cleanly with data (5% at 400
episodes -> 17.5% at 1200), while PPO fine-tuning collapsed that same
pretrained policy to 0% success almost immediately and never recovered --
even after fixing a genuine GRU hidden-state/importance-ratio bug in the PPO
loop. DAgger's own policy-mix rollout (vs. imitation_pretrain's fixed random
exploration) already addresses the covariate-shift problem PPO fine-tuning
was meant to solve here, without PPO's instability.

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
    def __init__(self, observation_size: int, action_size: int = 4, hidden_size: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh())
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.actor, self.hidden_size = nn.Linear(hidden_size, action_size), hidden_size

    def forward_sequence(self, observations, episode_starts, initial_hidden=None):
        hidden = (torch.zeros(1, 1, self.hidden_size, device=observations.device)
                  if initial_hidden is None else initial_hidden.clone())
        logits = []
        for t in range(observations.shape[0]):
            hidden = hidden * (1.0 - episode_starts[t].float().view(1, 1, 1))
            output, hidden = self.gru(self.encoder(observations[t]).view(1, 1, -1), hidden)
            logits.append(self.actor(output[0, 0]))
        return torch.stack(logits)

    @torch.no_grad()
    def greedy_action(self, observation, hidden, episode_start):
        if episode_start:
            hidden.zero_()
        output, hidden = self.gru(self.encoder(observation).view(1, 1, -1), hidden)
        return self.actor(output[0, 0]).argmax().item(), hidden


@torch.no_grad()
def evaluate(policy: RecurrentPolicy, env: MIMORFNavigationEnv, device: torch.device,
             episodes: int = 60, split: str = "train") -> tuple[float, float]:
    """Deterministic (greedy) success rate and mean episode reward.

    split="holdout" starts only from cells never drawn during split="train"
    rollouts, so it measures generalisation to unseen states rather than
    just an unlucky random draw from the same trained-on distribution.
    """
    policy.eval()
    successes, reward_sum = 0, 0.0
    for _ in range(episodes):
        observation, done, episode_start = env.reset(split=split), False, True
        hidden = torch.zeros(1, 1, policy.hidden_size, device=device)
        while not done:
            action, hidden = policy.greedy_action(
                torch.as_tensor(observation, device=device), hidden, episode_start
            )
            observation, reward, done, info = env.step(action)
            reward_sum += reward
            episode_start = done
        successes += int(info["reached"])
    policy.train()
    return successes / episodes, reward_sum / episodes


def collect_round(policy: RecurrentPolicy, env: MIMORFNavigationEnv, episodes: int,
                   device: torch.device, beta: float):
    """Roll out a per-step beta-mix of teacher/policy actions.

    Every visited state is labelled with the teacher action regardless of
    which action was actually executed -- the state distribution comes
    increasingly from the learner's own (possibly imperfect) choices as beta
    decays across rounds, which is what lets DAgger correct states a fixed
    demonstration set would never cover, without PPO's instability.
    """
    demonstrations = []
    successes, reward_sum = 0, 0.0
    for _ in range(episodes):
        observation, done = env.reset(), False
        hidden = torch.zeros(1, 1, policy.hidden_size, device=device)
        observations, targets = [], []
        while not done:
            teacher_action = env.expert_action()
            if env.rng.random() < beta:
                executed = teacher_action
            else:
                with torch.no_grad():
                    output, hidden = policy.gru(
                        policy.encoder(torch.as_tensor(observation, device=device)).view(1, 1, -1), hidden
                    )
                    executed = policy.actor(output[0, 0]).argmax().item()
            observations.append(observation)
            targets.append(teacher_action)
            observation, reward, done, info = env.step(executed)
            reward_sum += reward
        demonstrations.append((np.asarray(observations, dtype=np.float32),
                               np.asarray(targets, dtype=np.int64)))
        successes += int(info["reached"])
    return demonstrations, successes / episodes, reward_sum / episodes


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
            logits = policy.forward_sequence(obs, starts)
            loss = nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 0.5); optimizer.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--episodes-per-round", type=int, default=150)
    parser.add_argument("--epochs-per-round", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=8, help="Episodes per gradient step.")
    parser.add_argument("--beta0", type=float, default=1.0, help="Round-1 probability of executing the teacher action.")
    parser.add_argument("--beta-decay", type=float, default=0.65, help="Multiplicative beta decay per round.")
    parser.add_argument("--beta-min", type=float, default=0.05)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MIMORFNavigationEnv(EnvConfig(), args.seed)
    policy = RecurrentPolicy(env.observation_size, env.action_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    Path("checkpoints").mkdir(exist_ok=True)

    all_demonstrations, best_success = [], -1.0
    beta = args.beta0
    for round_idx in range(1, args.rounds + 1):
        demos, rollout_success, rollout_reward = collect_round(policy, env, args.episodes_per_round, device, beta)
        all_demonstrations.extend(demos)
        loss = train_on_dataset(policy, optimizer, all_demonstrations, device, args.epochs_per_round, args.group_size)
        det_success, det_reward = evaluate(policy, MIMORFNavigationEnv(seed=args.seed + round_idx), device, args.eval_episodes)
        print(f"round={round_idx:>2}/{args.rounds} beta={beta:.2f} "
              f"dataset={sum(len(t) for _, t in all_demonstrations):>7} "
              f"rollout-success={rollout_success:.1%} rollout-reward={rollout_reward:6.2f} "
              f"eval-success={det_success:.1%} eval-reward={det_reward:6.2f} loss={loss:.3f}")
        if det_success >= best_success:
            best_success = det_success
            torch.save({"model": policy.state_dict(), "observation_size": env.observation_size,
                       "env_config": asdict(env.config)}, "checkpoints/rf_gru_dagger.pt")
        beta = max(args.beta_min, beta * args.beta_decay)

    print(f"Best deterministic success={best_success:.1%}; saved checkpoints/rf_gru_dagger.pt")


if __name__ == "__main__":
    main()
