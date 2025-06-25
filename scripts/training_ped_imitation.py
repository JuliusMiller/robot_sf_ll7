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


def unwrap_array(obj):
    if isinstance(obj, DictObs):
        return {k: unwrap_array(v) for k, v in obj.items()}
    else:
        return np.asarray(obj)


def only_one_setup_data(json_file: str, obs_space):
    with open(json_file) as f:
        data = json.load(f)

    num_timesteps = len(data["states"])
    print(f"Number of timesteps: {num_timesteps}")

    drive_shape = obs_space["drive_state"].shape
    rays_shape = obs_space["rays"].shape

    ped_idx = 0
    obs = []
    acts = []

    positions = [data["states"][t]["pedestrian_positions"][ped_idx] for t in range(num_timesteps)]
    positions = np.array(positions, dtype=np.float32)

    accelerations, steering_angles = compute_accel_and_steering(positions, time_interval=0.1)

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

    # Add final obs (len(obs) = len(acts) + 1)
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

    # Debug print to verify shapes/types
    print(f"obs type: {type(obs[0])}, obs keys: {list(obs[0].keys())}")
    print(f"acts shape: {acts.shape}, acts dtype: {acts.dtype}")

    rews = np.zeros(len(acts), dtype=np.float32)

    # Helper to unwrap DictObs safely
    def unwrap_dictobs(o):
        if isinstance(o, DictObs):
            return {k: np.asarray(v) for k, v in o._d.items()}
        elif isinstance(o, dict):
            return {k: np.asarray(v) for k, v in o.items()}
        else:
            return np.asarray(o)

    # Stack obs arrays properly to avoid IndexError
    stacked_obs = {}
    for key in obs[0].keys():
        arrays = []
        for o in obs:
            unwrapped = unwrap_dictobs(o)
            arrays.append(unwrapped[key])
        stacked_obs[key] = np.stack(arrays, axis=0)
    dict_obs = DictObs(stacked_obs)

    trajectory = TrajectoryWithRew(
        obs=dict_obs,
        acts=acts,
        rews=rews,
        infos=None,
        terminal=True,
    )

    print("✅ Successfully created 1 expert trajectory.")
    return [trajectory]


def unwrap_dictobs(o):
    if isinstance(o, DictObs):
        return {k: np.asarray(v) for k, v in o._d.items()}
    elif isinstance(o, dict):
        return {k: np.asarray(v) for k, v in o.items()}
    else:
        return np.asarray(o)


def setup_data(json_file: str, obs_space):
    with open(json_file) as f:
        data = json.load(f)

    num_timesteps = len(data["states"])
    num_pedestrians = len(data["states"][0]["pedestrian_positions"])
    print(f"Number of timesteps: {num_timesteps}, Number of pedestrians: {num_pedestrians}")

    drive_shape = obs_space["drive_state"].shape
    rays_shape = obs_space["rays"].shape

    trajectories = []

    for ped_idx in range(num_pedestrians):
        obs = []
        acts = []

        positions = [
            data["states"][t]["pedestrian_positions"][ped_idx] for t in range(num_timesteps)
        ]
        positions = np.array(positions, dtype=np.float32)

        accelerations, steering_angles = compute_accel_and_steering(positions, time_interval=0.1)

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

        # Final observation (len(obs) = len(acts) + 1)
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

        # Stack observations into DictObs
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
        print(f"✅ Trajectory for pedestrian {ped_idx} created.")

    print(f"✅ Successfully created {len(trajectories)} expert trajectories.")
    return trajectories


def training(svg_map_path: str, dataset_file: str):
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

    env = DummyVecEnv([lambda: Monitor(RolloutInfoWrapper(make_env()))])

    trajectories = setup_data(dataset_file, env.envs[0].observation_space)

    print(f"Number of trajectories: {len(trajectories)}")
    for i, traj in enumerate(trajectories):
        print(f"Trajectory {i}: type={type(traj)}")
        print(f"  obs type: {type(traj.obs)}, acts type: {type(traj.acts)}")
        if hasattr(traj, "obs"):
            print(f"  first obs type: {type(traj.obs[0])}")

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=trajectories,
        rng=rng,
    )

    reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print(f"Reward before training: {reward_before_training}")

    bc_trainer.train(n_epochs=1)
    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print(f"Reward after training: {reward_after_training}")


if __name__ == "__main__":
    SVG_MAP = "maps/svg_maps/debug_06.svg"
    JSON_FILE = "robot_sf/data_analysis/datasets/2025-06-25_13-44-10.json"

    def demo():
        demo = setup_data(JSON_FILE)
        print(f"Number of demonstrations: {len(demo)}")
        print(f"First observation shape: {demo[0].obs.shape}")
        print(f"First action shape: {demo[0].acts.shape}")
        print(f"First observation: {demo[0].obs[0]}")
        print(f"First action: {demo[0].acts[0]}")
        print(f"Second observation: {demo[0].obs[1]}")

    training(SVG_MAP, JSON_FILE)

    # training(SVG_MAP)
