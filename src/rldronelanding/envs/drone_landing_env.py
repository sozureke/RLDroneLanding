import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class DroneLandingEnv(gym.Env):
	def __init__(self, render_mode = None) -> None:
		super().__init__()
		self.render_mode = render_mode
		self.time_step = 1.0 / 60.0
		self.max_steps = 500
		self.current_step = 0
		self.space_limit = 10.0
		self.platform_start = [0, 0, -0.5]
		self.platform_surface_z = self.platform_start[2] + 1.0

		self.observation_space = spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(10,),
			dtype=np.float32
		)

		self.action_space = spaces.Box(
			low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
			high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
			dtype=np.float32
		)

		self.client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setTimeStep(self.time_step)
		p.setGravity(0, 0, -9.81)

		self.drone = None
		self.platform = None

	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		self.current_step = 0
		p.resetSimulation()
		p.setGravity(0, 0, -9.81)
		p.loadURDF("plane.urdf")

		x = np.random.uniform(-5, 5)
		y = np.random.uniform(-5, 5)
		z = np.random.uniform(4, 8)

		self.drone = p.loadURDF("sphere_small.urdf", [x, y, z])
		self.platform = p.loadURDF("cube.urdf", self.platform_start, globalScaling=2.0, useFixedBase=True)

		obs = self._get_obs()
		return obs, {}

	def step(self, action):
		self.current_step += 1
		force = np.clip(action, -1.0, 1.0) * 10.0
		p.applyExternalForce(self.drone, -1, force, [0, 0, 0], p.WORLD_FRAME)
		p.stepSimulation()

		if self.render_mode == "human":
			time.sleep(self.time_step)

		obs = self._get_obs()
		reward, terminated, truncated = self._compute_reward(obs)

		return obs, reward, terminated, truncated, {}

	def _get_obs(self):
		drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
		drone_vel, _ = p.getBaseVelocity(self.drone)
		platform_pos, _ = p.getBasePositionAndOrientation(self.platform)

		rel_pos = np.array(drone_pos) - np.array(platform_pos)
		obs = np.concatenate([
			rel_pos,  # dx, dy, dz
			drone_vel,  # vx, vy, vz
			drone_pos[:2],  # drone x, y
			platform_pos[:2]  # platform x, y
		])

		return obs.astype(np.float32)

	def _compute_reward(self, obs):
		drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
		platform_pos, _ = p.getBasePositionAndOrientation(self.platform)

		dx = abs(drone_pos[0] - platform_pos[0])
		dy = abs(drone_pos[1] - platform_pos[1])
		dz = abs(drone_pos[2] - self.platform_surface_z)

		velocity = obs[3:6]
		speed = np.linalg.norm(velocity)

		distance = np.linalg.norm([
			drone_pos[0] - platform_pos[0],
			drone_pos[1] - platform_pos[1],
			drone_pos[2] - self.platform_surface_z
		])

		landed = dx < 0.5 and dy < 0.5 and dz < 0.25 and speed < 0.5

		out_of_bounds = abs(obs[2]) > self.space_limit
		max_steps_reached = self.current_step >= self.max_steps

		reward = -0.05 * distance - 0.01 * speed

		if landed:
			reward += 50
			print(f"[SUCCESS] Landed! dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, speed={speed:.2f}")

		if out_of_bounds and not landed:
			reward -= 100
			print(f"[FAIL] Drone crashed or flew out of bounds. z={obs[2]:.2f}")

		terminated = landed
		truncated = out_of_bounds or max_steps_reached
		return reward, terminated, truncated

	def render(self):
		pass

	def close(self):
		p.disconnect(self.client)



