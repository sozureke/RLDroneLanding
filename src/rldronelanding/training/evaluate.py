import argparse
import numpy as np
from stable_baselines3 import PPO
from rldronelanding.envs.drone_landing_env import DroneLandingEnv
from rldronelanding.utils.visualization import plot_trajectories


def evaluate(model_path, episodes=5, render=True, save_trajectories=False):
    env = DroneLandingEnv(render_mode="human" if render else None)
    model = PPO.load(model_path)

    total_rewards = []
    all_trajectories = []

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        ep_reward = 0
        trajectory = []

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()

            if save_trajectories:
                pos = obs[6:8]  # drone x, y
                trajectory.append(pos)

        print(f"\n🎬 Episode {ep + 1} finished")
        print(f"→ Total Reward: {ep_reward:.2f}")
        print(f"→ Final Position: x={obs[6]:.2f}, y={obs[7]:.2f}, z≈{obs[2]:.2f}")

        if terminated:
            print("[LANDING] ✅ Agent landed successfully!")
        elif truncated:
            print("[FAIL] ⛔ Timeout or out-of-bounds.")

        total_rewards.append(ep_reward)

        if save_trajectories:
            all_trajectories.append(np.array(trajectory))

    env.close()

    print("\n📊 Evaluation Summary:")
    print(f"→ Episodes: {episodes}")
    print(f"→ Average Reward: {np.mean(total_rewards):.2f}")
    print(f"→ Std Deviation: {np.std(total_rewards):.2f}")

    if save_trajectories:
        plot_trajectories(all_trajectories)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="models/ppo_drone.zip", help="Path to trained model"
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--save-trajectories", action="store_true")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        render=not args.no_render,
        save_trajectories=args.save_trajectories
    )
