import argparse
import numpy as np
from stable_baselines3 import PPO
from rldronelanding.envs.drone_landing_env import DroneLandingEnv
from rldronelanding.utils.visualization import plot_trajectories

def evaluate(model_path, episodes=5, render=True, save_trajectories=False,
             wind=False, moving_platform=False, platform_speed=1.0,
             sensor_noise=0.0, drone_scale=1.5):
    env = DroneLandingEnv(
        render_mode="human" if render else None,
        enable_wind=wind,
        enable_platform_motion=moving_platform,
        platform_speed=platform_speed,
        sensor_noise_std=sensor_noise,
        drone_scale=drone_scale
    )
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
                pos = obs[6:8]
                trajectory.append(pos)

        print(f"\n🎬 Episode {ep + 1}")
        print(f"→ Total Reward: {ep_reward:.2f}")
        print(f"→ Final Position: x={obs[6]:.2f}, y={obs[7]:.2f}, z≈{obs[2]:.2f}")

        if terminated:
            print("[LANDING] Successful landing.")
        elif truncated:
            print("[FAIL] Timeout or out-of-bounds.")

        total_rewards.append(ep_reward)
        if save_trajectories:
            all_trajectories.append(np.array(trajectory))

    env.close()

    print("\n📊 Evaluation Summary:")
    print(f"→ Episodes: {episodes}")
    print(f"→ Avg Reward: {np.mean(total_rewards):.2f}")
    print(f"→ Std Dev: {np.std(total_rewards):.2f}")

    if save_trajectories:
        plot_trajectories(all_trajectories)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_drone.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--wind", action="store_true")
    parser.add_argument("--moving-platform", action="store_true")
    parser.add_argument("--platform-speed", type=float, default=1.0)
    parser.add_argument("--sensor-noise", type=float, default=0.0)
    parser.add_argument("--drone-scale", type=float, default=1.0)

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        render=not args.no_render,
        save_trajectories=args.save_trajectories,
        wind=args.wind,
        moving_platform=args.moving_platform,
        platform_speed=args.platform_speed,
        sensor_noise=args.sensor_noise,
        drone_scale=args.drone_scale
    )
