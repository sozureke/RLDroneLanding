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


def make_env(render: bool = False):
  def _init():
    env = DroneLandingEnv(render_mode="human" if render else None)
    env = Monitor(env)
    return env

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

    args = parser.parse_args()
    config = load_config(args.config)

    env = DummyVecEnv([make_env(render=config["render"])])
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
    print("✅ The training is complete. The model has been saved.")

    env.close()

if __name__ == "__main__":
  main()
