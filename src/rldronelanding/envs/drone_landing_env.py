import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
from robot_descriptions import cf2_description

class DroneLandingEnv(gym.Env):
    def __init__(
        self,
        render_mode=None,
        enable_wind=False,
        enable_platform_motion=False,
        platform_speed=0.0,
        sensor_noise_std=0.0,
        drone_scale=1.5
    ):
        super().__init__()
        self.render_mode = render_mode
        self.time_step = 1.0 / 60.0
        self.max_steps = 500
        self.current_step = 0
        self.space_limit = 10.0

        self.platform_start = [0, 0, -0.5]
        self.platform_surface_z = self.platform_start[2] + 1.0
        self.platform_speed = platform_speed

        self.enable_wind = enable_wind
        self.enable_platform_motion = enable_platform_motion
        self.wind_force_range = [-2.0, 2.0]
        self.sensor_noise_std = sensor_noise_std

        self.drone_scale = drone_scale
        self.max_thrust = 20.0
        self.max_pitch_angle = np.deg2rad(10)
        self.max_roll_angle = np.deg2rad(10)
        self.max_yaw_angle = np.deg2rad(10)

        self.mass = 1.0
        self.g = 9.81
        self.hover_factor = 1.0
        self.hover_thrust = self.mass * self.g * self.hover_factor

        self.kp_z_hover = 3.0
        self.kd_z_hover = 2.0
        self.kp_z_approach = 4.0
        self.kd_z_approach = 2.5
        self.kp_z_landing = 6.0
        self.kd_z_landing = 3.0

        self.ki_z = 0.02
        self.max_vertical_integral = 3.0
        self.vertical_integral = 0.0
        self.last_vertical_error = 0.0
        self.target_altitude = None

        self.kp_pitch = 8.0
        self.kd_pitch = 3.0
        self.ki_pitch = 0.2
        self.kp_roll = 5.0
        self.kd_roll = 2.5
        self.kp_yaw = 5.0
        self.kd_yaw = 0.5

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        obs_limit = np.array([20.0] * 15, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_limit, high=obs_limit, dtype=np.float32)

        self.client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, -9.81)

        self.drone = None
        self.platform = None
        self.freeze_drone = False
        self.last_obs = None
        self.current_state = "hover"
        self.approach_dist = 2.0
        self.landing_dist = 0.8
        self.stable_landing_steps = 0
        self.required_stable_steps = 3

        if render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.freeze_drone = False
        self.vertical_integral = 0.0
        self.last_vertical_error = 0.0
        self.current_state = "hover"
        self.stable_landing_steps = 0

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        y = np.random.uniform(-1.5, 1.5)
        z = np.random.uniform(3, 4)
        drone_start_pos = [0, y, z]
        try:
            drone_urdf_path = cf2_description.URDF_PATH
            orientation = p.getQuaternionFromEuler([0, 0, 0])
            self.drone = p.loadURDF(
                fileName=drone_urdf_path,
                basePosition=drone_start_pos,
                baseOrientation=orientation,
                useFixedBase=False,
                globalScaling=self.drone_scale
            )
            p.changeDynamics(
                self.drone, -1,
                mass=self.mass,
                lateralFriction=1.5,
                spinningFriction=0.3,
                rollingFriction=0.1,
                linearDamping=0.2,
                angularDamping=0.5,
                restitution=0.0
            )
        except Exception as e:
            print(f"[ERROR] Loading drone URDF failed: {e}. Using sphere fallback.")
            self.drone = p.loadURDF("sphere_small.urdf", drone_start_pos, globalScaling=self.drone_scale)

        self.platform = p.loadURDF("cube.urdf", [0, 0, -0.5], globalScaling=2.0, useFixedBase=True)
        p.changeDynamics(self.platform, -1, restitution=0.0, lateralFriction=1.0)
        self.target_altitude = self.platform_surface_z

        obs = self._get_obs()
        self.last_obs = obs
        return obs, {}

    def step(self, action):
        if self.freeze_drone:
            return self.last_obs, 0.0, True, False, {}

        self.current_step += 1

        desired_pitch = np.clip(action[0], -1, 1) * self.max_pitch_angle
        desired_roll = np.clip(action[1], -1, 1) * self.max_roll_angle
        desired_yaw = np.clip(action[2], -1, 1) * self.max_yaw_angle
        user_thrust_factor = np.clip(action[3], -1.0, 1.0)

        obs = self._get_obs()
        self._update_state(obs)

        drone_pos, drone_orn = p.getBasePositionAndOrientation(self.drone)
        euler = p.getEulerFromQuaternion(drone_orn)
        current_roll, current_pitch, current_yaw = euler
        _, ang_vel = p.getBaseVelocity(self.drone)

        altitude_error = self.target_altitude - drone_pos[2]
        self.vertical_integral += altitude_error * self.time_step
        if self.vertical_integral > self.max_vertical_integral:
            self.vertical_integral = self.max_vertical_integral
        elif self.vertical_integral < -self.max_vertical_integral:
            self.vertical_integral = -self.max_vertical_integral

        vertical_derivative = (altitude_error - self.last_vertical_error) / self.time_step
        self.last_vertical_error = altitude_error

        kp_z, kd_z = self._select_pid_gains()
        vertical_pid = kp_z * altitude_error + self.ki_z * self.vertical_integral + kd_z * vertical_derivative

        net_thrust = self.hover_thrust + vertical_pid
        user_thrust_adjustment = user_thrust_factor * 4.0
        net_thrust += user_thrust_adjustment
        net_thrust = np.clip(net_thrust, 0.0, self.max_thrust)

        error_pitch = desired_pitch - current_pitch
        error_roll = desired_roll - current_roll
        error_yaw = desired_yaw - current_yaw
        torque_pitch = self.kp_pitch * error_pitch - self.kd_pitch * ang_vel[1]
        torque_roll = self.kp_roll * error_roll - self.kd_roll * ang_vel[0]
        torque_yaw = self.kp_yaw * error_yaw - self.kd_yaw * ang_vel[2]

        p.applyExternalForce(self.drone, -1, [0, 0, net_thrust], [0, 0, 0], p.WORLD_FRAME)
        p.applyExternalTorque(self.drone, -1, [torque_roll, torque_pitch, torque_yaw], p.WORLD_FRAME)

        if self.enable_wind:
            wind_x = np.random.uniform(*self.wind_force_range)
            wind_y = np.random.uniform(*self.wind_force_range)
            p.applyExternalForce(self.drone, -1, [wind_x, wind_y, 0], [0, 0, 0], p.WORLD_FRAME)

        if self.enable_platform_motion:
            t = self.current_step * self.time_step * self.platform_speed
            new_x = self.platform_start[0] + np.sin(t) * 0.5
            p.resetBasePositionAndOrientation(
                self.platform,
                [new_x, self.platform_start[1], self.platform_start[2]],
                [0, 0, 0, 1]
            )

        p.stepSimulation()

        if self.render_mode == "human":
            time.sleep(self.time_step * 1.5)

        obs = self._get_obs()
        self.last_obs = obs
        reward, landed, truncated = self._compute_reward(obs)
        if landed:
            p.resetBaseVelocity(self.drone, [0, 0, 0], [0, 0, 0])
            self.freeze_drone = True
            self.last_obs = obs

        if self.current_step >= self.max_steps or abs(drone_pos[2]) > self.space_limit:
            truncated = True
        return obs, reward, landed, truncated, {}

    def _update_state(self, obs):
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        dx = drone_pos[0]
        dy = drone_pos[1]
        dist_xy = np.sqrt(dx**2 + dy**2)
        z = drone_pos[2]

        if self.current_state == "hover":
            if dist_xy > self.approach_dist or z > (self.platform_surface_z + 0.5):
                self.current_state = "approach"
        elif self.current_state == "approach":
            if dist_xy < self.landing_dist and z < (self.platform_surface_z + 1.0):
                self.current_state = "landing"
        elif self.current_state == "landing":
            pass

    def _select_pid_gains(self):
        if self.current_state == "hover":
            return self.kp_z, self.kd_z
        elif self.current_state == "approach":
            return self.kp_z_approach, self.kd_z_approach
        elif self.current_state == "landing":
            return self.kp_z_landing, self.kd_z_landing
        return self.kp_z, self.kd_z

    def _get_obs(self):
        drone_pos, drone_orn = p.getBasePositionAndOrientation(self.drone)
        drone_vel, ang_vel = p.getBaseVelocity(self.drone)
        platform_pos, _ = p.getBasePositionAndOrientation(self.platform)
        rel_pos = np.array(drone_pos) - np.array(platform_pos)
        euler_angles = p.getEulerFromQuaternion(drone_orn)
        obs = np.concatenate([
            rel_pos,
            drone_vel,
            euler_angles,
            ang_vel,
            platform_pos[:2],
            [self.platform_surface_z]
        ])
        if self.sensor_noise_std > 0:
            noise = np.random.normal(0, self.sensor_noise_std, size=obs.shape)
            obs += noise
        return obs.astype(np.float32)

    def _compute_reward(self, obs):
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        platform_pos, _ = p.getBasePositionAndOrientation(self.platform)
        dx = drone_pos[0] - platform_pos[0]
        dy = drone_pos[1] - platform_pos[1]
        dz = drone_pos[2] - self.platform_surface_z
        horizontal_distance = np.sqrt(dx**2 + dy**2)
        vertical_error = abs(dz)
        drone_vel, _ = p.getBaseVelocity(self.drone)
        speed = np.linalg.norm(drone_vel)
        vz = drone_vel[2]
        roll, pitch, _ = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.drone)[1])

        distance_reward = -0.1 * horizontal_distance - 0.2 * vertical_error
        speed_penalty = -0.05 * speed - 0.1 * abs(vz)
        orientation_penalty = -0.02 * (abs(roll) + abs(pitch))

        contact_points = p.getContactPoints(self.drone, self.platform)
        is_touching = len(contact_points) > 0

        landing_bonus = 0
        gentle_landing = False
        if horizontal_distance < 0.4 and vertical_error < 0.15 and abs(vz) < 0.1 and abs(roll) < 0.2 and abs(pitch) < 0.2 and is_touching:
            self.stable_landing_steps += 1
        else:
            self.stable_landing_steps = 0
        if self.stable_landing_steps >= self.required_stable_steps:
            gentle_landing = True
            landing_bonus = 150

        out_of_bounds_penalty = 0
        if abs(drone_pos[2]) > self.space_limit:
            out_of_bounds_penalty = -50

        total_reward = distance_reward + speed_penalty + orientation_penalty + landing_bonus + out_of_bounds_penalty
        terminated = gentle_landing
        truncated = False
        return total_reward, terminated, truncated

    def set_platform_speed(self, speed: float):
        self.platform_speed = speed

    def set_sensor_noise(self, noise_std: float):
        self.sensor_noise_std = noise_std

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)