import gymnasium as gym
import yaml
import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from rldronelanding.envs.drone_landing_env import DroneLandingEnv


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_env(render: bool = False, wind: bool = False, platform_motion: bool = False):
    def _init():
        env = DroneLandingEnv(
            render_mode="human" if render else None,
            enable_wind=wind,
            enable_platform_motion=platform_motion
        )
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="src/rldronelanding/config/training_config.yaml"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="Total number of training timesteps",
    )
    parser.add_argument("--wind", action="store_true", help="Enable wind force")
    parser.add_argument("--moving-platform", action="store_true", help="Enable moving landing platform")

    args = parser.parse_args()
    config = load_config(args.config)

    wind_enabled = args.wind or config.get("enable_wind", False)
    platform_motion_enabled = args.moving_platform or config.get("enable_platform_motion", False)

    print("🚀 Training configuration:")
    print(f"→ Timesteps: {args.timesteps}")
    print(f"→ Wind Enabled: {'Yes' if args.wind else 'No'}")
    print(f"→ Moving Platform Enabled: {'Yes' if args.moving_platform else 'No'}")
    print()

    env = DummyVecEnv([
        make_env(
            render=config.get("render", False),
            wind=wind_enabled,
            platform_motion=platform_motion_enabled
        )
    ])
    env = VecMonitor(env)

    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=config["model_save_path"],
        name_prefix="ppo_checkpoint"
    )

    model = PPO(
        policy=config["policy"],
        env=env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        ent_coef=config["ent_coef"],
        clip_range=config["clip_range"],
        gae_lambda=config["gae_lambda"],
        verbose=1,
        tensorboard_log=config["log_dir"]
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback
    )

    model.save(config["model_save_path"])
    print("The training is complete. The model has been saved.")

    env.close()


if __name__ == "__main__":
    main()
