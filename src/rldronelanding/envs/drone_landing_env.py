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
        enable_wind: bool = False,
        enable_platform_motion: bool = False,
        platform_speed: float = 1.0,
        sensor_noise_std: float = 0.0,
        drone_scale: float = 1.0
    ):
        super().__init__()
        self.render_mode = render_mode
        self.time_step = 1.0 / 60.0
        self.max_steps = 500
        self.current_step = 0
        self.step_counter = 0
        self.space_limit = 10.0

        # PLATFORM PARAMETERS
        self.platform_start = [0, 0, -0.5]
        self.platform_surface_z = self.platform_start[2] + 1.0
        self.platform_speed = platform_speed

        # ENVIRONMENT PARAMETERS
        self.enable_wind = enable_wind
        self.enable_platform_motion = enable_platform_motion
        self.wind_force_range = [-2.0, 2.0]
        self.sensor_noise_std = sensor_noise_std

        # DRONE AND STABILIZER PARAMETERS
        self.drone_scale = drone_scale
        self.max_thrust = 20.0  # максимальная тяга
        self.max_lateral_force = 6.0
        self.max_pitch_angle = np.deg2rad(10)
        self.max_roll_angle = np.deg2rad(10)
        self.max_yaw_angle = np.deg2rad(10)

        # PID коэффициенты

        self.kp_pitch = 8.0
        self.kd_pitch = 3.0
        self.ki_pitch = 0.2
        self.kp_roll = 5.0
        self.kd_roll = 2.5
        self.kp_yaw = 5.0
        self.kd_yaw = 0.5



        # ACTION SPACE: [desired_pitch, desired_roll, desired_yaw, desired_thrust]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
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

    def reset(self, seed=None, options=None):
        ideal_pitch = 0.1
        super().reset(seed=seed)
        self.current_step = 0
        self.step_counter = 0
        self.freeze_drone = False

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")


        y = np.random.uniform(-1.5, 1.5)
        z = np.random.uniform(3, 4)

        drone_start_pos = [0, y, z]

        try:
            drone_urdf_path = cf2_description.URDF_PATH
            # random_offset = np.random.uniform(-0.1, 0.1)
            orientation = p.getQuaternionFromEuler([0, 0, 0])
            self.drone = p.loadURDF(
                fileName=drone_urdf_path,
                basePosition=drone_start_pos,
                baseOrientation=orientation,
                useFixedBase=False,
                globalScaling=self.drone_scale
            )
            com_offset = [0, 0, -0.1]
            p.changeDynamics(self.drone, -1, localInertiaDiagonal=com_offset)


            # LINES
            if self.render_mode == "human":
                p.addUserDebugLine([0, 0, 0], [0.3, 0, 0], [1, 0, 0], parentObjectUniqueId=self.drone)
                p.addUserDebugLine([0, 0, 0], [0, 0.3, 0], [0, 1, 0], parentObjectUniqueId=self.drone)
                p.addUserDebugLine([0, 0, 0], [0, 0, 0.3], [0, 0, 1], parentObjectUniqueId=self.drone)


        except Exception as e:
            print(f"[ERROR] Loading drone URDF failed: {e}. Using sphere fallback.")
            self.drone = p.loadURDF("sphere_small.urdf", drone_start_pos, globalScaling=1.2)

        pos, orn = p.getBasePositionAndOrientation(self.drone)
        euler = p.getEulerFromQuaternion(orn)  # [roll, pitch, yaw]
        if euler[1] < (ideal_pitch - 0.2) or euler[1] > (ideal_pitch + 0.2):
            corrected_orientation = p.getQuaternionFromEuler([0, ideal_pitch, 0])
            p.resetBasePositionAndOrientation(self.drone, drone_start_pos, corrected_orientation)
            print(f"[INFO] Drone orientation corrected to ideal pitch {ideal_pitch} rad.")

        self.platform = p.loadURDF("cube.urdf", self.platform_start, globalScaling=2.0, useFixedBase=True)
        p.changeDynamics(self.platform, -1, restitution=0.0, lateralFriction=1.0)
        if self.drone is not None:
            p.changeDynamics(
                self.drone,
                -1,
                mass=1.0,
                lateralFriction=1.5,
                spinningFriction=0.3,
                rollingFriction=0.1,
                linearDamping=0.2,
                angularDamping=0.5,
                restitution=0.0
            )

        obs = self._get_obs()
        self.last_obs = obs
        return obs, {}

    def step(self, action):
        if self.freeze_drone:
            return self.last_obs, 0, True, False, {}

        roll, pitch, yaw = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.drone)[1])
        with open("drone_angles.log", "a") as f:
            f.write(f"{self.step_counter},{roll},{pitch},{yaw}\n")

        self.current_step += 1
        self.step_counter += 1

        desired_pitch = np.clip(action[0], -1, 1) * self.max_pitch_angle
        desired_roll = np.clip(action[1], -1, 1) * self.max_roll_angle
        desired_yaw = np.clip(action[2], -1, 1) * self.max_yaw_angle
        desired_thrust = np.clip(action[3], 0, 1) * self.max_thrust

        pos, orn = p.getBasePositionAndOrientation(self.drone)
        euler = p.getEulerFromQuaternion(orn)  # [roll, pitch, yaw]
        current_roll, current_pitch, current_yaw = euler
        _, ang_vel = p.getBaseVelocity(self.drone)

        error_pitch = desired_pitch - current_pitch
        error_roll = desired_roll - current_roll
        error_yaw = desired_yaw - current_yaw

        torque_pitch = self.kp_pitch * error_pitch - self.kd_pitch * ang_vel[1]
        torque_roll = self.kp_roll * error_roll - self.kd_roll * ang_vel[0]
        torque_yaw = self.kp_yaw * error_yaw - self.kd_yaw * ang_vel[2]

        p.applyExternalTorque(self.drone, -1, [torque_roll, torque_pitch, torque_yaw], p.WORLD_FRAME)
        p.applyExternalForce(self.drone, -1, [0, 0, desired_thrust], [0, 0, 0], p.WORLD_FRAME)

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
            time.sleep(self.time_step)

        obs = self._get_obs()
        self.last_obs = obs
        reward, landed, termination_condition = self._compute_reward(obs)
        if landed:
            p.resetBaseVelocity(self.drone, [0, 0, 0], [0, 0, 0])
            self.freeze_drone = True
            self.last_obs = obs

        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        dz = abs(drone_pos[2] - self.platform_surface_z)
        lin_vel, _ = p.getBaseVelocity(self.drone)
        speed = np.linalg.norm(lin_vel)
        if dz < 0.3 and speed > 0:
            damp_factor = 0.5
            new_lin_vel = [v * damp_factor for v in lin_vel]
            p.resetBaseVelocity(self.drone, new_lin_vel, [0, 0, 0])

        return obs, reward, landed, termination_condition, {}

    def _get_obs(self):
        drone_pos, drone_orn = p.getBasePositionAndOrientation(self.drone)
        drone_vel, ang_vel = p.getBaseVelocity(self.drone)
        platform_pos, _ = p.getBasePositionAndOrientation(self.platform)
        rel_pos = np.array(drone_pos) - np.array(platform_pos)
        euler_angles = p.getEulerFromQuaternion(drone_orn)  # [roll, pitch, yaw]

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
        dx, dy, dz = drone_pos[0] - platform_pos[0], drone_pos[1] - platform_pos[1], drone_pos[
            2] - self.platform_surface_z
        horizontal_distance = np.sqrt(dx ** 2 + dy ** 2)
        vertical_error = abs(dz)

        lin_vel, _ = p.getBaseVelocity(self.drone)
        speed = np.linalg.norm(lin_vel)
        v_z = lin_vel[2]

        distance_reward = -0.1 * horizontal_distance - 0.2 * vertical_error
        speed_penalty = -0.05 * speed - 0.1 * abs(v_z)
        roll, pitch, _ = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.drone)[1])
        orientation_penalty = -0.02 * (abs(roll) + abs(pitch))
        contact_points = p.getContactPoints(self.drone, self.platform)
        is_touching = len(contact_points) > 0

        if (
            horizontal_distance < 0.6
            and vertical_error < 0.25
            and abs(v_z) < 0.15
            and abs(roll) < 0.3
            and abs(pitch) < 0.3
        ):
            landing_bonus = 100
            gentle_landing = True
        else:
            landing_bonus = 0
            gentle_landing = False

        out_of_bounds_penalty = -50 if abs(drone_pos[2]) > self.space_limit else 0
        total_reward = distance_reward + speed_penalty + orientation_penalty + landing_bonus + out_of_bounds_penalty

        terminated = gentle_landing
        truncated = (abs(drone_pos[2]) > self.space_limit) or (self.current_step >= self.max_steps)

        return total_reward, terminated, truncated

    def set_platform_speed(self, speed: float):
        self.platform_speed = speed

    def set_sensor_noise(self, noise_std: float):
        self.sensor_noise_std = noise_std

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)
