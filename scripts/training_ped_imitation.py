"""
Script for Behavioral Cloning (BC) imitation learning of pedestrian behavior.

- Loads expert pedestrian trajectories from JSON.
- Converts positions to acceleration and steering angle for use as actions.
- Prepares observations in the format expected by the environment.
- Trains a BC policy using the imitation library.
"""

import json

import loguru
import numpy as np
from imitation.algorithms import bc
from imitation.data.types import DictObs, TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from robot_sf.gym_env.env_config import PedEnvSettings
from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings

logger = loguru.logger


def compute_accel_and_steering(positions, time_interval=0.1):
    """
    Given an array of positions (N, 2), compute acceleration and steering angle arrays (N-2,).
    Returns:
        accelerations: (N-2,)
        steering_angles: (N-2,)
    """
    positions = np.asarray(positions)
    velocities = np.diff(positions, axis=0) / time_interval  # (N-1, 2)
    speeds = np.linalg.norm(velocities, axis=1)  # (N-1,)
    accelerations = np.diff(speeds) / time_interval  # (N-2,)
    headings = np.arctan2(velocities[:, 1], velocities[:, 0])  # (N-1,)
    steering_angles = np.diff(headings) / time_interval  # (N-2,)
    return accelerations, steering_angles


def unwrap_dictobs(o):
    """
    Helper to convert DictObs or dict to dict of np.ndarray.
    """
    if isinstance(o, DictObs):
        return {k: np.asarray(v) for k, v in o._d.items()}
    elif isinstance(o, dict):
        return {k: np.asarray(v) for k, v in o.items()}
    else:
        return np.asarray(o)


def setup_data(json_file: str, obs_space):
    """
    Loads expert pedestrian trajectories from a JSON file and converts them into
    a format suitable for imitation learning (acceleration and steering angle as actions).

    Args:
        json_file (str): Path to the JSON file with expert data.
        obs_space: Observation space of the environment.

    Returns:
        List[TrajectoryWithRew]: List of expert trajectories.
    """
    with open(json_file) as f:
        data = json.load(f)

    num_timesteps = len(data["states"])
    num_pedestrians = len(data["states"][0]["pedestrian_positions"])
    logger.info(f"Number of timesteps: {num_timesteps}, Number of pedestrians: {num_pedestrians}")

    drive_shape = obs_space["drive_state"].shape
    rays_shape = obs_space["rays"].shape

    trajectories = []

    for ped_idx in range(num_pedestrians):
        obs = []
        acts = []

        # Extract all positions for this pedestrian
        positions = [
            data["states"][t]["pedestrian_positions"][ped_idx] for t in range(num_timesteps)
        ]
        positions = np.array(positions, dtype=np.float32)

        # Compute acceleration and steering angle from positions
        accelerations, steering_angles = compute_accel_and_steering(positions, time_interval=0.1)

        # Build observations and actions for each valid timestep
        for t in range(num_timesteps - 2):
            drive_state = np.zeros(drive_shape, dtype=np.float32)
            drive_state[:, 0] = accelerations[t]
            drive_state[:, 1] = steering_angles[t]
            rays = np.zeros(rays_shape, dtype=np.float32)
            obs_dict = DictObs(
                {
                    "drive_state": drive_state.astype(np.float32),
                    "rays": rays.astype(np.float32),
                }
            )
            obs.append(obs_dict)
            acts.append(np.array([accelerations[t], steering_angles[t]], dtype=np.float32))

        # Add the final observation (no action after it)
        drive_state = np.zeros(drive_shape, dtype=np.float32)
        rays = np.zeros(rays_shape, dtype=np.float32)
        obs.append(
            DictObs(
                {
                    "drive_state": drive_state.astype(np.float32),
                    "rays": rays.astype(np.float32),
                }
            )
        )

        acts = np.array(acts, dtype=np.float32)
        rews = np.zeros(len(acts), dtype=np.float32)

        # Stack observations into DictObs (dict of arrays)
        stacked_obs = {}
        for key in obs[0].keys():
            arrays = [unwrap_dictobs(o)[key] for o in obs]
            stacked_obs[key] = np.stack(arrays, axis=0)
        dict_obs = DictObs(stacked_obs)

        trajectory = TrajectoryWithRew(
            obs=dict_obs,
            acts=acts,
            rews=rews,
            infos=None,
            terminal=True,
        )
        trajectories.append(trajectory)

    logger.info(f"Successfully created expert trajectories: {len(trajectories)}.")
    return trajectories


def training(svg_map_path: str, dataset_file: str):
    """
    Main training loop for behavioral cloning imitation learning.

    Args:
        svg_map_path (str): Path to SVG map file.
        dataset_file (str): Path to JSON file with expert demonstrations.
    """
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2
    rng = np.random.default_rng(42)

    def make_env():
        map_definition = convert_map(svg_map_path)
        robot_model = PPO.load("./model/run_043", env=None)

        env_config = PedEnvSettings(
            map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
            sim_config=SimulationSettings(
                difficulty=difficulty, ped_density_by_difficulty=ped_densities
            ),
            robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
        )

        env = PedestrianEnv(env_config, robot_model=robot_model)
        return env

    # Wrap environment for monitoring and info collection
    env = DummyVecEnv([lambda: Monitor(RolloutInfoWrapper(make_env()))])

    # Prepare expert trajectories
    trajectories = setup_data(dataset_file, env.envs[0].observation_space)

    # Initialize BC trainer
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=trajectories,
        rng=rng,
    )

    # Evaluate before training
    reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    logger.info(f"Reward before training: {reward_before_training}")

    # Train for 1 epoch (increase for real training)
    bc_trainer.train(n_epochs=1)

    # Evaluate after training
    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    logger.info(f"Reward after training: {reward_after_training}")


if __name__ == "__main__":
    SVG_MAP = "maps/svg_maps/debug_06.svg"
    JSON_FILE = "robot_sf/data_analysis/datasets/2025-06-25_13-44-10.json"

    # Run training with one expert trajectory
    training(SVG_MAP, JSON_FILE)
