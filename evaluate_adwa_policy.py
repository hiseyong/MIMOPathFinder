"""Evaluate a saved ADWA-benchmark policy and render one RF-only trajectory.

Defaults to a holdout building (see adwa_env.HOLDOUT_BUILDINGS): a real
floorplan never sampled during training, so this is a genuine
unseen-environment test rather than a replay of a memorised route.
"""
import argparse
import math
from pathlib import Path

import torch

from adwa_env import ADWANavigationEnv
from train_adwa_dagger import RecurrentPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("holdout", "train", "all"), default="holdout",
                        help="Building pool to draw the episode from; 'holdout' is never seen during training.")
    parser.add_argument("--seed", type=int, default=None, help="Omit for a different random building/start each run.")
    args = parser.parse_args()

    data = torch.load("checkpoints/adwa_gru_dagger.pt", map_location="cpu", weights_only=True)
    env = ADWANavigationEnv()
    policy = RecurrentPolicy(data["observation_size"])
    policy.load_state_dict(data["model"]); policy.eval()
    observation = env.reset(seed=args.seed, split=args.split)
    pos0 = (round(float(env.position[0]), 2), round(float(env.position[1]), 2))
    hidden, trajectory, done, episode_start = torch.zeros(1, 1, policy.hidden_size), [pos0], False, True
    print(f"building={env.building_name} (split={args.split}) source={env.source} grid={env.world.shape}")
    print(f"step   0: pos={pos0} heading={math.degrees(env.heading):.1f}deg")
    while not done:
        direction, hidden = policy.greedy_direction(torch.as_tensor(observation), hidden, episode_start)
        observation, reward, done, info = env.step(direction)
        pos = (round(float(env.position[0]), 2), round(float(env.position[1]), 2))
        trajectory.append(pos)
        angle_deg = math.degrees(math.atan2(direction[1], direction[0]))
        flag = "  <- COLLISION" if info["collision"] else ("  <- REACHED" if info["reached"] else "")
        print(f"step {info['steps']:3d}: pos={pos} dir={angle_deg:6.1f}deg reward={reward:+.2f}{flag}")
        episode_start = False
    Path("outputs").mkdir(exist_ok=True)
    env.render(trajectory).savefig("outputs/adwa_evaluation_trajectory.png", dpi=160, bbox_inches="tight")
    env.render_animation(trajectory).save("outputs/adwa_evaluation_trajectory.gif", writer="pillow", fps=8)
    print(f"Reached source: {info['reached']}; steps: {info['steps']}")
    print("Saved outputs/adwa_evaluation_trajectory.png (final route) "
          "and outputs/adwa_evaluation_trajectory.gif (step-by-step animation)")


if __name__ == "__main__":
    main()
