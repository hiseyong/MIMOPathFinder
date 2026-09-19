"""Evaluate a saved policy and render one RF-only trajectory.

Defaults to a holdout map: one of the layouts/source positions never sampled
during training (see mimo_rl_env.N_TRAIN_MAPS/N_HOLDOUT_MAPS), so this is a
genuine unseen-environment test rather than a replay of a memorised route.
"""
import argparse
from pathlib import Path

import torch

from mimo_rl_env import MIMORFNavigationEnv
from train_dagger import RecurrentPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("holdout", "train", "all"), default="holdout",
                        help="Map pool to draw the episode from; 'holdout' is never seen during training.")
    parser.add_argument("--seed", type=int, default=None, help="Omit for a different random map/start each run.")
    args = parser.parse_args()

    data = torch.load("checkpoints/rf_gru_dagger.pt", map_location="cpu", weights_only=True)
    env = MIMORFNavigationEnv()
    policy = RecurrentPolicy(data["observation_size"], env.action_size)
    policy.load_state_dict(data["model"]); policy.eval()
    observation = env.reset(seed=args.seed, split=args.split)
    hidden, trajectory, done, episode_start = torch.zeros(1, 1, policy.hidden_size), [env.position], False, True
    print(f"map_index={env.map_index} (split={args.split}) source={env.source}")
    print(f"step   0: pos={env.position} heading={env.heading}")
    while not done:
        action, hidden = policy.greedy_action(torch.as_tensor(observation), hidden, episode_start)
        observation, reward, done, info = env.step(action)
        pos = (int(env.position[0]), int(env.position[1]))
        trajectory.append(pos)
        flag = "  <- COLLISION" if info["collision"] else ("  <- REACHED" if info["reached"] else "")
        print(f"step {info['steps']:3d}: pos={pos} action={env.ACTION_NAMES[action]:<22} reward={reward:+.2f}{flag}")
        episode_start = False
    Path("outputs").mkdir(exist_ok=True)
    env.render(trajectory).savefig("outputs/evaluation_trajectory.png", dpi=160, bbox_inches="tight")
    env.render_animation(trajectory).save("outputs/evaluation_trajectory.gif", writer="pillow", fps=8)
    print(f"Reached source: {info['reached']}; steps: {info['steps']}")
    print("Saved outputs/evaluation_trajectory.png (final route) "
          "and outputs/evaluation_trajectory.gif (step-by-step animation)")


if __name__ == "__main__":
    main()
