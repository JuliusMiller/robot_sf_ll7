from math import ceil
from dataclasses import dataclass, field
from typing import Tuple, Union, List
from copy import deepcopy

import numpy as np
from gym import Env, spaces

from robot_sf.sim_config import MapDefinitionPool
from robot_sf.occupancy import ContinuousOccupancy
from robot_sf.range_sensor import ContinuousLidarScanner, LidarScannerSettings
from robot_sf.sim_view import SimulationView, VisualizableAction, VisualizableSimState
from robot_sf.robot import DifferentialDriveRobot, RobotSettings, rel_pos
from robot_sf.ped_robot_force import PedRobotForceConfig
from robot_sf.simulator import Simulator


Vec2D = Tuple[float, float]
PolarVec2D = Tuple[float, float]


@dataclass
class SimulationSettings:
    sim_length_in_secs: float = 200.0
    step_time_in_secs: float = 0.1
    peds_speed_mult: float = 1.3
    difficulty: int = 2
    max_peds_per_group: int = 6
    prf_config: PedRobotForceConfig = PedRobotForceConfig(is_active=True)
    ped_density_by_difficulty: List[float] = field(default_factory=lambda: [0.0, 0.02, 0.04, 0.06])

    @property
    def max_sim_steps(self) -> int:
        return ceil(self.sim_length_in_secs / self.step_time_in_secs)

    @property
    def peds_per_area_m2(self) -> float:
        return self.ped_density_by_difficulty[self.difficulty]


@dataclass
class EnvSettings:
    sim_config: SimulationSettings = SimulationSettings()
    lidar_config: LidarScannerSettings = LidarScannerSettings()
    robot_config: RobotSettings = RobotSettings()
    map_pool: MapDefinitionPool = MapDefinitionPool()


@dataclass
class EnvState:
    is_pedestrian_collision: bool
    is_obstacle_collision: bool
    is_robot_at_goal: bool
    is_timesteps_exceeded: bool

    @property
    def is_terminal(self) -> bool:
        return self.is_timesteps_exceeded or self.is_pedestrian_collision or \
            self.is_obstacle_collision or self.is_robot_at_goal


@dataclass
class SimpleReward:
    max_sim_steps: float
    step_discount: float = field(init=False)

    def __post_init__(self):
        self.step_discount = 0.1 / self.max_sim_steps

    def __call__(self, state: EnvState) -> float:
        reward = -self.step_discount
        if state.is_pedestrian_collision or state.is_obstacle_collision:
            reward -= 2
        if state.is_robot_at_goal:
            reward += 1
        return reward


class RobotEnv(Env):
    """Representing an OpenAI Gym environment wrapper for
    training a robot with reinforcement leanring"""

    def __init__(self, env_config: EnvSettings = EnvSettings(), debug: bool=False):
        self.sim_config = env_config.sim_config
        self.lidar_config = env_config.lidar_config
        self.robot_config = env_config.robot_config

        map_def = env_config.map_pool.choose_random_map()
        width, height = map_def.width, map_def.height

        self.env_type = 'RobotEnv'
        self.max_sim_steps = self.sim_config.max_sim_steps
        self.max_target_dist = np.sqrt(2) * (max(width, height) * 2) # the box diagonal
        self.action_space, self.observation_space = \
            RobotEnv._build_gym_spaces(self.max_target_dist, self.robot_config, self.lidar_config)
        self.reward_func = SimpleReward(self.max_sim_steps)

        self.sim_env: Simulator
        self.occupancy = ContinuousOccupancy(
            width, height,
            lambda: self.sim_env.robot_pose[0],
            lambda: self.sim_env.goal_pos,
            lambda: self.sim_env.pysf_sim.env.obstacles_raw,
            lambda: self.sim_env.current_positions,
            self.robot_config.radius)

        self.lidar_sensor = ContinuousLidarScanner(self.lidar_config, self.occupancy)
        robot_factory = lambda s, g: DifferentialDriveRobot(s, g, self.robot_config)
        self.sim_env = Simulator(self.sim_config, map_def, robot_factory)

        self.episode = 0
        self.timestep = 0
        self.last_action: Union[PolarVec2D, None] = None
        if debug:
            self.sim_ui = SimulationView()

    def step(self, action: np.ndarray):
        action_parsed = (action[0], action[1])
        self.sim_env.step_once(action_parsed)
        self.last_action = action_parsed
        obs = self._get_obs()
        state = EnvState(
            self.occupancy.is_pedestrian_collision,
            self.occupancy.is_obstacle_collision,
            self.occupancy.is_robot_at_goal,
            self.timestep > self.max_sim_steps)
        reward, done = self.reward_func(state), state.is_terminal
        self.timestep += 1
        return obs, reward, done, { 'step': self.episode, 'meta': state }

    def reset(self):
        self.episode += 1
        self.timestep = 0
        self.last_action = None
        self.sim_env.reset_state()
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        ranges_np = self.lidar_sensor.get_scan(self.sim_env.robot_pose)
        speed_x, speed_rot = self.sim_env.robot.current_speed
        target_distance, target_angle = rel_pos(self.sim_env.robot.pose, self.sim_env.goal_pos)

        # normalize observations within [0, 1] or [-1, 1]
        ranges_np /= self.lidar_config.max_scan_dist
        speed_x /= self.robot_config.max_linear_speed
        speed_rot = speed_rot / self.robot_config.max_angular_speed
        target_distance /= self.max_target_dist
        target_angle = target_angle / np.pi

        robot_state = np.array([speed_x, speed_rot, target_distance, target_angle])
        return np.concatenate((ranges_np, robot_state), axis=0)

    def render(self, mode='human'):
        if not self.sim_ui:
            raise RuntimeError('Debug mode is not activated! Consider setting debug=True!')

        action = None if not self.last_action else \
            VisualizableAction(self.sim_env.robot.pose, self.last_action, self.sim_env.goal_pos)

        state = VisualizableSimState(
            self.timestep,
            action,
            self.sim_env.robot.pose,
            deepcopy(self.occupancy.pedestrian_coords),
            deepcopy(self.occupancy.obstacle_coords))

        self.sim_ui.render(state)

    @staticmethod
    def _build_gym_spaces(
            max_target_dist: float, robot_config: RobotSettings, \
            lidar_config: LidarScannerSettings) -> Tuple[spaces.Box, spaces.Box]:
        action_low  = np.array([-robot_config.max_linear_speed, -robot_config.max_angular_speed])
        action_high = np.array([ robot_config.max_linear_speed,  robot_config.max_angular_speed])
        action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)

        state_max = np.concatenate((
                lidar_config.max_scan_dist * np.ones((lidar_config.scan_length,)),
                np.array([robot_config.max_linear_speed, robot_config.max_angular_speed,
                          max_target_dist, np.pi])), axis=0)
        state_min = np.concatenate((
                np.zeros((lidar_config.scan_length,)),
                np.array([0, -robot_config.max_angular_speed, 0, -np.pi])
            ), axis=0)
        observation_space = spaces.Box(low=state_min, high=state_max, dtype=np.float64)
        return action_space, observation_space
