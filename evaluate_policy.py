"""Evaluate a saved policy and render one RF-only trajectory."""
from pathlib import Path

import torch

from mimo_rl_env import MIMORFNavigationEnv
from train_dagger import RecurrentPolicy


def main():
    data = torch.load("checkpoints/rf_gru_dagger.pt", map_location="cpu", weights_only=True)
    env = MIMORFNavigationEnv()
    policy = RecurrentPolicy(data["observation_size"], env.action_size)
    policy.load_state_dict(data["model"]); policy.eval()
    observation, hidden, trajectory, done = env.reset(random_start=False), torch.zeros(1, 1, policy.hidden_size), [env.position], False
    episode_start = True
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
