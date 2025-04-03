import yaml
import os
import argparse


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from rldronelanding.envs.drone_landing_env import DroneLandingEnv


class ProgressiveDifficultyCallback(BaseCallback):
    def __init__(
        self,
        env,
        interval: int = 20000,
        speed_increment: float = 0.2,
        max_speed: float = 2.5,
        noise_increment: float = 0.01,
        max_noise: float = 0.3,
        verbose: int = 0
    ) -> None:
        super().__init__(verbose)
        self.env = env
        self.interval = interval
        self.speed_increment = speed_increment
        self.max_speed = max_speed
        self.noise_increment = noise_increment
        self.max_noise = max_noise
        self.current_speed = None
        self.current_noise = None

    def _on_training_start(self) -> None:
        self.current_speed = self.env.get_attr("platform_speed")[0]
        self.current_noise = self.env.get_attr("sensor_noise_std")[0]

    def _on_step(self) -> bool:
        if self.n_calls % self.interval == 0:
            # Increase platform speed
            new_speed = min(self.current_speed + self.speed_increment, self.max_speed)
            self.env.env_method("set_platform_speed", speed=new_speed)
            self.current_speed = new_speed

            # Increase sensor noise
            new_noise = min(self.current_noise + self.noise_increment, self.max_noise)
            self.env.env_method("set_sensor_noise", noise_std=new_noise)
            self.current_noise = new_noise

            print(f"\n Difficulty increased! Platform: {new_speed:.1f}m/s, Noise: {new_noise:.2f}")
        return True

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def make_env(render=False, wind=False, platform_motion=False, platform_speed=1.0, sensor_noise=0.0, drone_scale=1.0):
    def _init():
        env = DroneLandingEnv(
            render_mode="human" if render else None,
            enable_wind=wind,
            enable_platform_motion=platform_motion,
            platform_speed=platform_speed,
            sensor_noise_std=sensor_noise,
            drone_scale=drone_scale
        )
        return env
    return _init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/rldronelanding/config/training_config.yaml")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--wind", action="store_true")
    parser.add_argument("--moving-platform", action="store_true")
    parser.add_argument("--platform-speed", type=float, default=1.0)
    parser.add_argument("--sensor-noise", type=float, default=0.0)
    parser.add_argument("--drone-scale", type=float, default=1.0)

    args = parser.parse_args()
    config = load_config(args.config)

    env = DummyVecEnv([
        make_env(
            render=config["render"],
            wind=args.wind,
            platform_motion=args.moving_platform,
            platform_speed=args.platform_speed,
            sensor_noise=args.sensor_noise,
            drone_scale=args.drone_scale
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

    difficulty_callback = ProgressiveDifficultyCallback(
        env=env,
        interval=20000,
        speed_increment=0.2,
        max_speed=2.5,
        noise_increment=0.01,
        max_noise=0.2
    )

    callbacks = CallbackList([checkpoint_callback, difficulty_callback])

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
        callback=callbacks,
        progress_bar=True
    )

    model.save(config["model_save_path"])
    print("✅ Training complete. Model saved.")
    env.close()

if __name__ == "__main__":
    main()
