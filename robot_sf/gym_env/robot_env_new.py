from copy import deepcopy
from typing import Callable

import loguru

from robot_sf.gym_env.abstract_envs import SingleAgentEnv
from robot_sf.gym_env.env_util import (
    init_collision_and_sensors,
    init_spaces,
    prepare_pedestrian_actions,
)
from robot_sf.gym_env.reward import simple_reward
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.render.lidar_visual import render_lidar
from robot_sf.render.sim_view import SimulationView, VisualizableAction, VisualizableSimState
from robot_sf.robot.robot_state import RobotState
from robot_sf.sensor.range_sensor import lidar_ray_scan
from robot_sf.sim.simulator import init_simulators

logger = loguru.logger


class NewRobotEnv(SingleAgentEnv):
    def __init__(
        self,
        env_config: RobotSimulationConfig = None,
        reward_func: Callable[[dict], float] = simple_reward,
        debug: bool = False,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: str = None,
        video_fps: float = None,
        peds_have_obstacle_forces: bool = False,
        **kwargs,
    ):
        """
        Initialize the Pedestrian Environment.

        Args:
            config: Pedestrian simulation configuration
            robot_model: Pre-trained robot model for adversarial interaction
            reward_func: Reward function for pedestrian training
            debug: Enable debug mode with visualization
            recording_enabled: Enable state recording
            peds_have_obstacle_forces: Whether pedestrians exert obstacle forces
        """
        if env_config is None:
            logger.warning("No config provided, using default RobotSimulationConfig.")
            env_config = RobotSimulationConfig()

        # Store reward function
        if reward_func is None:
            logger.warning("No reward function provided, using default simple_reward.")
            self.reward_func = simple_reward
        else:
            self.reward_func = reward_func

        # Update config
        env_config.peds_have_obstacle_forces = peds_have_obstacle_forces

        # Initialize base class
        super().__init__(
            config=env_config, debug=debug, recording_enabled=recording_enabled, **kwargs
        )

    def _setup_environment(self) -> None:
        """Initialize environment-specific components."""
        # Get map definition
        self.map_def = self.config.map_pool.choose_random_map()

        # Initialize spaces
        self.action_space, self.observation_space, self.orig_obs_space = self._create_spaces()

        # Setup simulator and sensors
        self._setup_simulator()
        self._setup_sensors_and_collision()

        # Setup visualization if in debug mode
        if self.debug:
            self._setup_visualization()

    def _create_spaces(self):
        """Create action and observation spaces."""
        action_space, observation_space, orig_obs_space = init_spaces(self.config, self.map_def)
        logger.error(f"Action space: {action_space}")
        logger.error(f"Observation space: {observation_space}")
        logger.error(f"Original observation space: {orig_obs_space}")
        return action_space, observation_space, orig_obs_space

    def _setup_simulator(self) -> None:
        """Initialize the simulator."""
        self.simulator = init_simulators(
            self.config,
            self.map_def,
            random_start_pos=True,
            peds_have_obstacle_forces=self.config.peds_have_obstacle_forces,
        )[0]

    def _setup_sensors_and_collision(self) -> None:
        """Initialize sensors and collision detection."""
        occupancies, sensors = init_collision_and_sensors(
            self.simulator, self.config, self.orig_obs_space
        )

        # Setup robot state
        self.state = RobotState(
            nav=self.simulator.robot_navs[0],
            occupancy=occupancies[0],
            sensors=sensors[0],
            d_t=self.config.sim_config.time_per_step_in_secs,
            sim_time_limit=self.config.sim_config.sim_time_in_secs,
        )

        # Store last action executed by the robot
        self.last_action = None

    def _setup_visualization(self) -> None:
        """Setup visualization for debug mode."""
        self.sim_ui = SimulationView(
            map_def=self.map_def,
            obstacles=self.map_def.obstacles,
            robot_radius=self.config.robot_config.radius,
            ped_radius=self.config.sim_config.ped_radius,
            goal_radius=self.config.sim_config.goal_radius,
        )

    def step(self, action):
        """Execute one environment step."""
        # Process the action through the simulator
        action = self.simulator.robots[0].parse_action(action)

        # Perform simulation step
        self.simulator.step_once([action])
        # Get updated observation
        obs = self.state.step()
        # Fetch metadata about the current state
        reward_dict = self.state.meta_dict()
        # add the action space to dict
        reward_dict["action_space"] = self.action_space
        # add action to dict
        reward_dict["action"] = action
        # Add last_action to reward_dict
        reward_dict["last_action"] = self.last_action
        # Determine if the episode has reached terminal state
        term = self.state.is_terminal
        # Compute the reward using the provided reward function
        reward = self.reward_func(reward_dict)
        # Update last_action for next step
        self.last_action = action

        # if recording is enabled, record the state
        if self.recording_enabled:
            self.record()

        # observation, reward, terminal, truncated,info
        return (
            obs,
            reward,
            term,
            False,
            {"step": reward_dict["step"], "meta": reward_dict},
        )

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed, options=options)

        # Reset last_action
        self.last_action = None

        # Reset simulator
        self.simulator.reset_state()

        # Reset state
        obs = self.state.reset()

        if self.recording_enabled:
            self.save_recording()

        return obs, {}

    def render(self):
        """
        Render the environment visually if in debug mode.

        Raises RuntimeError if debug mode is not enabled.
        """
        if not self.sim_ui:
            raise RuntimeError("Debug mode is not activated! Consider setting `debug=True!`")

        state = self._prepare_visualizable_state()

        # Execute rendering of the state through the simulation UI
        self.sim_ui.render(state)

    def _prepare_visualizable_state(self):
        # Prepare action visualization, if any action was executed
        action = (
            None
            if not self.last_action
            else VisualizableAction(
                self.simulator.robot_poses[0],
                self.last_action,
                self.simulator.goal_pos[0],
            )
        )

        # Robot position and LIDAR scanning visualization preparation
        robot_pos = self.simulator.robot_poses[0][0]
        distances, directions = lidar_ray_scan(
            self.simulator.robot_poses[0],
            self.state.occupancy,
            self.config.lidar_config,
        )

        # Construct ray vectors for visualization
        ray_vecs_np = render_lidar(robot_pos, distances, directions)

        # Prepare pedestrian action visualization
        ped_actions_np = prepare_pedestrian_actions(self.simulator)

        # Package the state for visualization
        state = VisualizableSimState(
            timestep=self.state.timestep,
            robot_action=action,
            robot_pose=self.simulator.robot_poses[0],
            pedestrian_positions=deepcopy(self.simulator.ped_pos),
            ray_vecs=ray_vecs_np,
            ped_actions=ped_actions_np,
            time_per_step_in_secs=self.config.sim_config.time_per_step_in_secs,
        )

        return state

    def record(self):
        """
        Records the current state as visualizable state and stores it in the list.
        """
        state = self._prepare_visualizable_state()
        self.recorded_states.append(state)

    def set_pedestrian_velocity_scale(self, scale: float = 1.0):
        """
        Set the pedestrian velocity visualization scaling factor.

        Args:
            scale (float): Scaling factor for pedestrian velocity arrows in visualization.
                          1.0 = actual size, 2.0 = double size for better visibility, etc.
        """
        if self.sim_ui:
            self.sim_ui.ped_velocity_scale = scale
        else:
            logger.warning("Cannot set velocity scale: debug mode not enabled")
