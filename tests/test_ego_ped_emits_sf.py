from typing import List

import numpy as np
import pytest
from stable_baselines3 import PPO

from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.gym_env.unified_config import PedEnvSettings
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.nav.nav_types import SvgRectangle
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.util.types import Line2D


@pytest.fixture
def dummy_map():
    # Minimal map with no obstacles, one route
    width = 40.0
    height = 40.0
    bounds: List[Line2D] = [
        (0, width, 0, 0),  # bottom
        (0, width, height, height),  # top
        (0, 0, 0, height),  # left
        (width, width, 0, height),  # right
    ]
    r_spawn = SvgRectangle(
        x=20, y=30, width=1, height=1, label="robot_spawn", id_="robot_spawn"
    ).get_zone()

    r_goal = SvgRectangle(
        x=30, y=30, width=1, height=1, label="robot_goal", id_="robot_goal"
    ).get_zone()

    r_path = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[[22, 30], [29, 30]],
        spawn_zone=r_spawn,
        goal_zone=r_goal,
    )

    ped_spawn = SvgRectangle(
        x=5, y=5, width=1, height=1, label="ped_spawn", id_="ped_spawn"
    ).get_zone()

    ped_goal = SvgRectangle(
        x=35, y=5, width=1, height=1, label="robot_goal", id_="robot_goal"
    ).get_zone()

    ped_path = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[[7, 5], [34, 5]],
        spawn_zone=ped_spawn,
        goal_zone=ped_goal,
    )

    map_def = MapDefinition(
        width=width,
        height=height,
        obstacles=[],
        robot_spawn_zones=[r_spawn],
        ped_spawn_zones=[ped_spawn],
        robot_goal_zones=[r_goal],
        bounds=bounds,
        robot_routes=[r_path],
        ped_goal_zones=[ped_goal],
        ped_crowded_zones=[],
        ped_routes=[ped_path],
    )
    return map_def


@pytest.fixture
def env(dummy_map):
    env_config = PedEnvSettings(
        map_pool=MapDefinitionPool(map_defs={"test": dummy_map}),
        sim_config=SimulationSettings(),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
    )

    robot_model = "./model/run_043"

    robot_model = PPO.load(robot_model, env=None)

    env = PedestrianEnv(
        env_config,
        robot_model=robot_model,
        debug=True,
        recording_enabled=False,
        peds_have_obstacle_forces=False,
    )
    return env


def test_npc_pedestrian_avoids_ego_pedestrian_social_force(env):
    """
    Test that the NPC pedestrian avoids the ego pedestrian using social forces.
    """
    obs, _ = env.reset()

    # Place ego at (10, 5), which is along the NPC's route from (7, 5) to (34, 5)
    env.simulator.ego_ped.state.pose = ((12.0, 5.0), 0)
    for i in range(200):
        action = np.array([0, 0])  # No action for the ego pedestrian
        obs, _, done, _, _ = env.step(action)
        # env.render()
        # time.sleep(0.1)

        if done:
            meta = env.ped_state.meta_dict()
            is_collision = meta.get("is_pedestrian_collision", False)
            assert not is_collision, "NPC collided with the ego pedestrian"
            obs, _ = env.reset()

    env.exit()


def test_ego_position_correct_in_states(env):
    """
    Test that the NPC pedestrian avoids the ego pedestrian using social forces.
    """
    obs, _ = env.reset()

    # Place ego at (10, 5), which is along the NPC's route from (7, 5) to (34, 5)
    for i in range(20):
        action = np.array([0, 0])  # No action for the ego pedestrian
        obs, _, done, _, _ = env.step(action)

        assert np.allclose(
            env.simulator.pysf_state.pysf_states()[-1, 0:2], env.simulator.ego_ped_pos
        ), "Ego pedestrian position does not match the simulator state"

    env.exit()
