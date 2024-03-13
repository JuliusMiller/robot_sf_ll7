"""
`robot_env.py` is a module that defines the simulation environment for a robot or multiple robots.
It includes classes and protocols for defining the robot's state, actions, and 
observations within the environment. 

Key components of this module include:

1. `Robot`: A protocol that outlines the necessary properties and methods a robot should have. 
These include observation space, action space, and methods to apply actions,
reset state, and parse actions.

2. `RobotState`: A data class that represents the state of a robot in the simulation environment.
It includes information about navigation, occupancy (for collision detection),
sensor fusion, and simulation time. It also tracks various conditions such as collision states,
timeout condition, simulation time elapsed, and timestep count.

3. `RobotEnv`: A class that represents the robot's environment. It inherits from `VectorEnv`
from the `gymnasium` library, which is a base class for environments that operate over
vectorized actions and observations. It includes methods for stepping through the environment,
resetting it, rendering it, and closing it.
It also defines the action and observation spaces for the robot.

4. `MultiRobotEnv`: A class that extends `RobotEnv` to handle multiple robots in the environment.
It overrides the `step_async` method to apply actions to all robots in the environment.
"""

from math import ceil
from typing import Tuple, Callable, List, Protocol, Any
from copy import deepcopy
from multiprocessing.pool import ThreadPool

import numpy as np
from gymnasium.vector import VectorEnv
from gymnasium import Env, spaces
from gymnasium.utils import seeding
from robot_sf.nav.map_config import MapDefinition

from robot_sf.robot.robot_state import RobotState
from robot_sf.sim_config import EnvSettings
from robot_sf.nav.occupancy import ContinuousOccupancy
from robot_sf.sensor.range_sensor import lidar_ray_scan, lidar_sensor_space
from robot_sf.sensor.goal_sensor import target_sensor_obs, target_sensor_space
from robot_sf.sensor.sensor_fusion import (
    fused_sensor_space, SensorFusion, OBS_RAYS, OBS_DRIVE_STATE)
from robot_sf.sim.sim_view import SimulationView, VisualizableAction, VisualizableSimState
from robot_sf.sim.simulator import Simulator


Vec2D = Tuple[float, float]
PolarVec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]


class Robot(Protocol):
    @property
    def observation_space(self) -> spaces.Box:
        raise NotImplementedError()

    @property
    def action_space(self) -> spaces.Box:
        raise NotImplementedError()

    @property
    def pos(self) -> Vec2D:
        raise NotImplementedError()

    @property
    def pose(self) -> RobotPose:
        raise NotImplementedError()

    @property
    def current_speed(self) -> PolarVec2D:
        raise NotImplementedError()

    def apply_action(self, action: Any, d_t: float):
        raise NotImplementedError()

    def reset_state(self, new_pose: RobotPose):
        raise NotImplementedError()

    def parse_action(self, action: Any) -> Any:
        raise NotImplementedError()


def simple_reward(
        meta: dict,
        max_episode_step_discount: float=-0.1,
        ped_coll_penalty: float=-5,
        obst_coll_penalty: float=-2,
        reach_waypoint_reward: float=1) -> float:
    """
    Calculate the reward for the robot's current state.

    Parameters:
    meta (dict): Metadata containing information about the robot's current state.
    max_episode_step_discount (float): Discount factor for each step in the episode.
    ped_coll_penalty (float): Penalty for colliding with a pedestrian.
    obst_coll_penalty (float): Penalty for colliding with an obstacle.
    reach_waypoint_reward (float): Reward for reaching a waypoint.

    Returns:
    float: The calculated reward.
    """

    # Initialize reward with a discount based on the maximum simulation steps
    reward = max_episode_step_discount / meta["max_sim_steps"]

    # If there's a collision with a pedestrian or another robot, apply penalty
    if meta["is_pedestrian_collision"] or meta["is_robot_collision"]:
        reward += ped_coll_penalty

    # If there's a collision with an obstacle, apply penalty
    if meta["is_obstacle_collision"]:
        reward += obst_coll_penalty

    # If the robot has reached its goal, apply reward
    if meta["is_robot_at_goal"]:
        reward += reach_waypoint_reward

    return reward


def init_simulators(
        env_config: EnvSettings,
        map_def: MapDefinition,
        num_robots: int = 1,
        random_start_pos: bool = True
        ) -> List[Simulator]:
    """
    Initialize simulators for the robot environment.

    Parameters:
    env_config (EnvSettings): Configuration settings for the environment.
    map_def (MapDefinition): Definition of the map for the environment.
    num_robots (int): Number of robots in the environment.
    random_start_pos (bool): Whether to start the robots at random positions.

    Returns:
    List[Simulator]: A list of initialized Simulator objects.
    """

    # Calculate the number of simulators needed based on the number of robots and start positions
    num_sims = ceil(num_robots / map_def.num_start_pos)

    # Calculate the proximity to the goal based on the robot radius and goal radius
    goal_proximity = env_config.robot_config.radius + env_config.sim_config.goal_radius

    # Initialize an empty list to hold the simulators
    sims: List[Simulator] = []

    # Create the required number of simulators
    for i in range(num_sims):
        # Determine the number of robots for this simulator
        n = map_def.num_start_pos if i < num_sims - 1 \
            else max(1, num_robots % map_def.num_start_pos)

        # Create the robots for this simulator
        sim_robots = [env_config.robot_factory() for _ in range(n)]

        # Create the simulator with the robots and add it to the list
        sim = Simulator(
            env_config.sim_config, map_def, sim_robots,
            goal_proximity, random_start_pos)
        sims.append(sim)

    return sims


def init_collision_and_sensors(
        sim: Simulator, env_config: EnvSettings, orig_obs_space: spaces.Dict):
    """
    Initialize collision detection and sensor fusion for the robots in the simulator.

    Parameters:
    sim (Simulator): The simulator object.
    env_config (EnvSettings): Configuration settings for the environment.
    orig_obs_space (spaces.Dict): Original observation space.

    Returns:
    Tuple[List[ContinuousOccupancy], List[SensorFusion]]:
        A tuple containing a list of occupancy objects for collision detection
        and a list of sensor fusion objects for sensor data handling.
    """

    # Get the number of robots, simulation configuration,
    # robot configuration, and lidar configuration
    num_robots = len(sim.robots)
    sim_config = env_config.sim_config
    robot_config = env_config.robot_config
    lidar_config = env_config.lidar_config

    # Initialize occupancy objects for each robot for collision detection
    occupancies = [ContinuousOccupancy(
            sim.map_def.width, sim.map_def.height,
            lambda: sim.robot_pos[i], lambda: sim.goal_pos[i],
            lambda: sim.pysf_sim.env.obstacles_raw[:, :4], lambda: sim.ped_pos,
            robot_config.radius, sim_config.ped_radius, sim_config.goal_radius)
        for i in range(num_robots)]

    # Initialize sensor fusion objects for each robot for sensor data handling
    sensor_fusions: List[SensorFusion] = []
    for r_id in range(num_robots):
        # Define the ray sensor, target sensor, and speed sensor for each robot
        ray_sensor = lambda r_id=r_id: lidar_ray_scan(
            sim.robots[r_id].pose, occupancies[r_id], lidar_config)[0]
        target_sensor = lambda r_id=r_id: target_sensor_obs(
            sim.robots[r_id].pose, sim.goal_pos[r_id], sim.next_goal_pos[r_id])
        speed_sensor = lambda r_id=r_id: sim.robots[r_id].current_speed

        # Create the sensor fusion object and add it to the list
        sensor_fusions.append(SensorFusion(
            ray_sensor, speed_sensor, target_sensor,
            orig_obs_space, sim_config.use_next_goal))

    return occupancies, sensor_fusions


def init_spaces(env_config: EnvSettings, map_def: MapDefinition):
    """
    Initialize the action and observation spaces for the environment.

    This function creates action and observation space using the factory method
    provided in the environment
    configuration, and then uses the robot's action space and observation space as the
    basis for the environment's action and observation spaces. The observation space is
    further extended with additional sensors.

    Parameters
    ----------
    env_config : EnvSettings
        The configuration settings for the environment.
    map_def : MapDefinition
        The definition of the map for the environment.

    Returns
    -------
    Tuple[Space, Space, Space]
        A tuple containing the action space, the extended observation space, and the
        original observation space of the robot.
    """
    # Create a robot using the factory method in the environment configuration
    robot = env_config.robot_factory()
    # Get the action space from the robot
    action_space = robot.action_space

    # Extend the robot's observation space with additional sensors
    observation_space, orig_obs_space = fused_sensor_space(
        env_config.sim_config.stack_steps,
        robot.observation_space,
        target_sensor_space(map_def.max_target_dist),
        lidar_sensor_space(
            env_config.lidar_config.num_rays,
            env_config.lidar_config.max_scan_dist)
        )

    # Return the action space, the extended observation space, and the original
    # observation space
    return action_space, observation_space, orig_obs_space


class RobotEnv(Env):
    """
    Representing a Gymnasium environment for training a self-driving robot
    with reinforcement learning.
    """

    def __init__(
            self,
            env_config: EnvSettings = EnvSettings(),
            reward_func: Callable[[dict], float] = simple_reward,
            debug: bool = False
            ):
        """
        Initialize the Robot Environment.

        Parameters:
        - env_config (EnvSettings): Configuration for environment settings.
        - reward_func (Callable[[dict], float]): Reward function that takes
            a dictionary as input and returns a float as reward.
        - debug (bool): If True, enables debugging information such as 
            visualizations.
        """

        # Environment configuration details
        self.env_config = env_config
        # Extract first map definition; currently only supports using the first map
        map_def = env_config.map_pool.map_defs[0]

        # Initialize spaces based on the environment configuration and map
        self.action_space, self.observation_space, orig_obs_space = \
            init_spaces(env_config, map_def)

        # Assign the reward function and debug flag
        self.reward_func, self.debug = reward_func, debug

        # Initialize simulator with a random start position
        self.simulator = init_simulators(
            env_config,
            map_def,
            random_start_pos=True
            )[0]

        # Delta time per simulation step and maximum episode time
        d_t = env_config.sim_config.time_per_step_in_secs
        max_ep_time = env_config.sim_config.sim_time_in_secs

        # Initialize collision detectors and sensor data processors
        occupancies, sensors = init_collision_and_sensors(
            self.simulator,
            env_config,
            orig_obs_space
            )

        # Setup initial state of the robot
        self.state = RobotState(
            self.simulator.robot_navs[0],
            occupancies[0],
            sensors[0],
            d_t,
            max_ep_time)

        # Store last action executed by the robot
        self.last_action = None

        # If in debug mode, create a simulation view to visualize the state
        if debug:
            self.sim_ui = SimulationView(
                scaling=10,
                obstacles=map_def.obstacles,
                robot_radius=env_config.robot_config.radius,
                ped_radius=env_config.sim_config.ped_radius,
                goal_radius=env_config.sim_config.goal_radius)

            # Display the simulation UI
            self.sim_ui.show()

    def step(self, action):
        """
        Execute one time step within the environment.

        Parameters:
        - action: Action to be executed.

        Returns:
        - obs: Observation after taking the action.
        - reward: Calculated reward for the taken action.
        - term: Boolean indicating if the episode has terminated.
        - info: Additional information as dictionary.
        """
        # Process the action through the simulator
        action = self.simulator.robots[0].parse_action(action)
        self.last_action = action
        # Perform simulation step
        self.simulator.step_once([action])
        # Get updated observation
        obs = self.state.step()
        # Fetch metadata about the current state
        meta = self.state.meta_dict()
        # Determine if the episode has reached terminal state
        term = self.state.is_terminal
        # Compute the reward using the provided reward function
        reward = self.reward_func(meta)
        return obs, reward, term, {"step": meta["step"], "meta": meta}

    def reset(self):
        """
        Reset the environment state to start a new episode.

        Returns:
        - obs: The initial observation after resetting the environment.
        """
        # Reset internal simulator state
        self.simulator.reset_state()
        # Reset the environment's state and return the initial observation
        obs = self.state.reset()
        return obs

    def render(self):
        """
        Render the environment visually if in debug mode.

        Raises RuntimeError if debug mode is not enabled.
        """
        if not self.sim_ui:
            raise RuntimeError(
                'Debug mode is not activated! Consider setting '
                'debug=True!')

        # Prepare action visualization, if any action was executed
        action = None if not self.last_action else VisualizableAction(
            self.simulator.robot_poses[0], self.last_action, 
            self.simulator.goal_pos[0])

        # Robot position and LIDAR scanning visualization preparation
        robot_pos = self.simulator.robot_poses[0][0]
        distances, directions = lidar_ray_scan(
            self.simulator.robot_poses[0], self.state.occupancy,
            self.env_config.lidar_config)

        # Construct ray vectors for visualization
        ray_vecs = zip(np.cos(directions) * distances, np.sin(directions) * distances)
        ray_vecs_np = np.array([[
            [robot_pos[0], robot_pos[1]],
            [robot_pos[0] + x, robot_pos[1] + y]
            ] for x, y in ray_vecs])

        # Prepare pedestrian action visualization
        ped_actions = zip(
            self.simulator.pysf_sim.peds.pos(),
            self.simulator.pysf_sim.peds.pos() +
            self.simulator.pysf_sim.peds.vel() * 2)
        ped_actions_np = np.array([[pos, vel] for pos, vel in ped_actions])

        # Package the state for visualization
        state = VisualizableSimState(
            self.state.timestep, action, self.simulator.robot_poses[0],
            deepcopy(self.simulator.ped_pos), ray_vecs_np, ped_actions_np)

        # Execute rendering of the state through the simulation UI
        self.sim_ui.render(state)

    def seed(self, seed=None):
        """
        Set the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
            number generators. The first value in the list should be the
            "main" seed, or the value which a reproducer should pass to
            'seed'. Often, the main seed equals the provided 'seed', but
            this won't be true if seed=None, for example.

        TODO: validate this method
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def exit(self):
        """
        Clean up and exit the simulation UI, if it exists.
        """
        if self.sim_ui:
            self.sim_ui.exit()


class MultiRobotEnv(VectorEnv):
    """Representing a Gymnasium environment for training
    multiple self-driving robots with reinforcement learning"""

    def __init__(
            self, env_config: EnvSettings = EnvSettings(),
            reward_func: Callable[[dict], float] = simple_reward,
            debug: bool = False, num_robots: int = 1):

        map_def = env_config.map_pool.map_defs[0] # info: only use first map
        action_space, observation_space, orig_obs_space = init_spaces(env_config, map_def)
        super(MultiRobotEnv, self).__init__(num_robots, observation_space, action_space)
        self.action_space = spaces.Box(
            low=np.array([self.single_action_space.low for _ in range(num_robots)]),
            high=np.array([self.single_action_space.high for _ in range(num_robots)]),
            dtype=self.single_action_space.low.dtype)

        self.reward_func, self.debug = reward_func, debug
        self.simulators = init_simulators(env_config, map_def, num_robots, random_start_pos=False)
        self.states: List[RobotState] = []
        d_t = env_config.sim_config.time_per_step_in_secs
        max_ep_time = env_config.sim_config.sim_time_in_secs

        for sim in self.simulators:
            occupancies, sensors = init_collision_and_sensors(sim, env_config, orig_obs_space)
            states = [
                RobotState(nav, occ, sen, d_t, max_ep_time)
                for nav, occ, sen in zip(sim.robot_navs, occupancies, sensors)
                ]
            self.states.extend(states)

        self.sim_worker_pool = ThreadPool(len(self.simulators))
        self.obs_worker_pool = ThreadPool(num_robots)

    def step(self, actions):
        actions = [self.simulators[0].robots[0].parse_action(a) for a in actions]
        i = 0
        actions_per_simulator = []
        for sim in self.simulators:
            num_robots = len(sim.robots)
            actions_per_simulator.append(actions[i:i+num_robots])
            i += num_robots

        self.sim_worker_pool.map(
            lambda s_a: s_a[0].step_once(s_a[1]),
            zip(self.simulators, actions_per_simulator))

        obs = self.obs_worker_pool.map(lambda s: s.step(), self.states)

        metas = [state.meta_dict() for state in self.states]
        masked_metas = [{ "step": meta["step"], "meta": meta } for meta in metas]
        masked_metas = (*masked_metas,)
        terms = [state.is_terminal for state in self.states]
        rewards = [self.reward_func(meta) for meta in metas]

        for i, (sim, state, term) in enumerate(zip(self.simulators, self.states, terms)):
            if term:
                sim.reset_state()
                obs[i] = state.reset()

        obs = { OBS_DRIVE_STATE: np.array([o[OBS_DRIVE_STATE] for o in obs]),
                OBS_RAYS: np.array([o[OBS_RAYS] for o in obs])}

        return obs, rewards, terms, masked_metas

    def reset(self):
        self.sim_worker_pool.map(lambda sim: sim.reset_state(), self.simulators)
        obs = self.obs_worker_pool.map(lambda s: s.reset(), self.states)

        obs = { OBS_DRIVE_STATE: np.array([o[OBS_DRIVE_STATE] for o in obs]),
                OBS_RAYS: np.array([o[OBS_RAYS] for o in obs]) }
        return obs

    def render(self, robot_id: int=0):
        # TODO: add support for PyGame rendering
        pass

    def close_extras(self, **kwargs):
        self.sim_worker_pool.close()
        self.obs_worker_pool.close()
