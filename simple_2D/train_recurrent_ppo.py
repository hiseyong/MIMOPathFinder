"""Train a GRU-PPO policy with only RF measurements and RF history.

DEPRECATED: measured on this codebase, PPO fine-tuning collapses the
DAgger-pretrained policy's success rate to 0% almost immediately and never
recovers, even after fixing a real GRU hidden-state/importance-ratio bug
(forward_sequence used to zero-init hidden state every rollout window
regardless of whether the window started mid-episode, corrupting the PPO
ratio against most rollouts since rollout=512 >> the ~160-step episode
length). Meanwhile pure DAgger pretraining scaled cleanly with data (5%
success at 400 episodes -> 17.5% at 1200). Use train_dagger.py instead.

BROKEN as of the continuous-direction action space and multi-map pool: this
file still assumes 4 discrete grid actions (env.expert_action,
env.action_size, env.ACTION_NAMES) which mimo_rl_env.MIMORFNavigationEnv no
longer has. Kept only as a record of the PPO-collapse finding above; do not
run it without porting it to the continuous API in train_dagger.py first.

Example: poetry run python train_recurrent_ppo.py --timesteps 100000
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from mimo_rl_env import EnvConfig, MIMORFNavigationEnv


class RecurrentActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int = 4, hidden_size: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh())
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.actor, self.critic, self.hidden_size = nn.Linear(hidden_size, action_size), nn.Linear(hidden_size, 1), hidden_size

    def forward_sequence(self, observations, episode_starts, initial_hidden=None):
        # initial_hidden must be the GRU state actually carried into this
        # window's first observation during rollout collection (act()). Any
        # window starting mid-episode that instead zero-inits here recomputes
        # log-probs/values from a different effective policy than collection
        # used, which corrupts the PPO importance ratio and destabilises
        # training even when the policy hasn't actually changed.
        hidden = (torch.zeros(1, 1, self.hidden_size, device=observations.device)
                  if initial_hidden is None else initial_hidden.clone())
        logits, values = [], []
        for t in range(observations.shape[0]):
            hidden *= 1.0 - episode_starts[t].float().view(1, 1, 1)
            output, hidden = self.gru(self.encoder(observations[t]).view(1, 1, -1), hidden)
            logits.append(self.actor(output[0, 0]))
            values.append(self.critic(output[0, 0]).squeeze())
        return torch.stack(logits), torch.stack(values)

    @torch.no_grad()
    def act(self, observation, hidden, episode_start):
        if episode_start:
            hidden.zero_()
        output, hidden = self.gru(self.encoder(observation).view(1, 1, -1), hidden)
        dist = Categorical(logits=self.actor(output[0, 0]))
        action = dist.sample()
        return action.item(), dist.log_prob(action), self.critic(output[0, 0]).squeeze(), hidden

    @torch.no_grad()
    def greedy_action(self, observation, hidden, episode_start):
        if episode_start:
            hidden.zero_()
        output, hidden = self.gru(self.encoder(observation).view(1, 1, -1), hidden)
        return self.actor(output[0, 0]).argmax().item(), hidden


def gae(rewards, values, dones, bootstrap, gamma=0.99, lam=0.95):
    advantages, carry = torch.zeros_like(rewards), torch.tensor(0.0)
    for t in reversed(range(len(rewards))):
        next_value = bootstrap if t == len(rewards) - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        carry = rewards[t] + gamma * next_value * not_done - values[t] + gamma * lam * not_done * carry
        advantages[t] = carry
    return advantages


def imitation_pretrain(policy, env, optimizer, episodes: int, device: torch.device,
                       exploration: float):
    """DAgger-style bootstrap from full trajectories, retaining GRU memory.

    A small fraction of executed actions is random, while the label remains the
    hidden A* recovery action. This supplies correction examples for RF states
    that the learned policy will visit after an imperfect decision.
    """
    demonstrations = []
    for _ in range(episodes):
        observation, done = env.reset(), False
        observations, targets = [], []
        while not done:
            action = env.expert_action()  # Not visible to the deployed policy.
            observations.append(observation)
            targets.append(action)
            executed = action if env.rng.random() >= exploration else int(env.rng.integers(env.action_size))
            observation, _, done, _ = env.step(executed)
        demonstrations.append((np.asarray(observations, dtype=np.float32),
                               np.asarray(targets, dtype=np.int64)))

    # Concatenating complete episodes while marking each boundary gives the GRU
    # action-conditioned RF history it will have at deployment time.
    sample_count = sum(len(targets) for _, targets in demonstrations)
    for _ in range(12):
        order = np.random.permutation(len(demonstrations))
        for indices in np.array_split(order, max(1, len(order) // 8)):
            sequences = [demonstrations[i] for i in indices]
            lengths = [len(targets) for _, targets in sequences]
            obs = torch.as_tensor(np.concatenate([features for features, _ in sequences]), device=device)
            labels = torch.as_tensor(np.concatenate([targets for _, targets in sequences]), device=device)
            starts = torch.zeros(len(labels), device=device)
            starts[np.cumsum([0, *lengths[:-1]])] = 1.0
            logits, _ = policy.forward_sequence(obs, starts)
            loss = nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    print(f"Imitation pretraining: {sample_count} A* RF/action samples; recovery exploration={exploration:.0%}")


@torch.no_grad()
def evaluate(policy, env, device: torch.device, episodes: int = 20) -> float:
    """Deterministic success rate; unlike PPO rollouts this has no sampling noise."""
    policy.eval()
    successes = 0
    for _ in range(episodes):
        observation, done, episode_start = env.reset(), False, True
        hidden = torch.zeros(1, 1, policy.hidden_size, device=device)
        while not done:
            action, hidden = policy.greedy_action(
                torch.as_tensor(observation, device=device), hidden, episode_start
            )
            observation, _, done, info = env.step(action)
            episode_start = done
        successes += int(info["reached"])
    policy.train()
    return successes / episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--rollout", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--imitation-episodes", type=int, default=400)
    parser.add_argument("--imitation-exploration", type=float, default=0.10)
    parser.add_argument("--bc-coef", type=float, default=0.20,
                        help="Keep PPO close to the RF/A* teacher during fine-tuning.")
    parser.add_argument("--value-coef", type=float, default=0.10)
    parser.add_argument("--eval-interval", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    env, device = MIMORFNavigationEnv(EnvConfig(), args.seed), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = RecurrentActorCritic(env.observation_size, env.action_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    Path("checkpoints").mkdir(exist_ok=True)
    if args.imitation_episodes:
        imitation_pretrain(policy, env, optimizer, args.imitation_episodes, device,
                           args.imitation_exploration)
        print(f"Pretrain deterministic success={evaluate(policy, MIMORFNavigationEnv(seed=args.seed + 1), device):.1%}")
    observation, hidden, episode_start = env.reset(), torch.zeros(1, 1, policy.hidden_size, device=device), True
    total_steps = completed = successes = 0
    recent_successes: deque[int] = deque(maxlen=50)

    while total_steps < args.timesteps:
        # The GRU state actually carried into this window's first observation
        # during collection; forward_sequence must be reseeded with it below,
        # not zero-initialised, since most windows start mid-episode.
        window_hidden = (torch.zeros_like(hidden) if episode_start else hidden).detach().clone()
        observations, actions, teacher_actions, log_probs, rewards, values, starts, dones = [], [], [], [], [], [], [], []
        for _ in range(min(args.rollout, args.timesteps - total_steps)):
            teacher_action = env.expert_action()  # Training-only stabilizer.
            action, log_prob, value, hidden = policy.act(torch.as_tensor(observation, device=device), hidden, episode_start)
            next_observation, reward, done, info = env.step(action)
            observations.append(observation); actions.append(action); teacher_actions.append(teacher_action); log_probs.append(log_prob.cpu()); rewards.append(reward)
            values.append(value.cpu()); starts.append(float(episode_start)); dones.append(float(done))
            total_steps += 1; observation, episode_start = next_observation, done
            if done:
                completed += 1; successes += int(info["reached"]); observation = env.reset()
                recent_successes.append(int(info["reached"]))

        obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device)
        act, teacher, old_lp = (torch.as_tensor(actions, device=device),
                                torch.as_tensor(teacher_actions, device=device),
                                torch.stack(log_probs).to(device))
        rew, old_values = torch.as_tensor(rewards, dtype=torch.float32, device=device), torch.stack(values).to(device)
        starts_t, dones_t = torch.as_tensor(starts, device=device), torch.as_tensor(dones, device=device)
        with torch.no_grad():
            bootstrap = torch.tensor(0.0) if episode_start else policy.act(torch.as_tensor(observation, device=device), hidden, False)[2].cpu()
            advantages = gae(rew.cpu(), old_values.cpu(), dones_t.cpu(), bootstrap).to(device)
            returns = advantages + old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(args.epochs):
            logits, predicted_values = policy.forward_sequence(obs, starts_t, initial_hidden=window_hidden)
            dist, new_lp = Categorical(logits=logits), Categorical(logits=logits).log_prob(act)
            ratio = torch.exp(new_lp - old_lp)
            policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages).mean()
            value_loss = 0.5 * (predicted_values - returns).pow(2).mean()
            entropy = dist.entropy().mean()
            imitation_loss = nn.functional.cross_entropy(logits, teacher)
            loss = (policy_loss + args.value_coef * value_loss - 0.001 * entropy
                    + args.bc_coef * imitation_loss)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 0.5); optimizer.step()
        deterministic = ""
        if total_steps % args.eval_interval < args.rollout:
            deterministic = f" eval={evaluate(policy, MIMORFNavigationEnv(seed=args.seed + total_steps), device):.1%}"
        recent = sum(recent_successes) / max(len(recent_successes), 1)
        print(f"steps={total_steps:>7} episodes={completed:>4} recent-success={recent:.1%}{deterministic} loss={loss.item():.3f}")

    torch.save({"model": policy.state_dict(), "observation_size": env.observation_size, "env_config": asdict(env.config)}, "checkpoints/rf_gru_ppo.pt")
    print("Saved checkpoints/rf_gru_ppo.pt")


if __name__ == "__main__":
    main()
