import loguru
from stable_baselines3 import PPO

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings

logger = loguru.logger


def make_env(svg_map_path):
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2

    map_definition = convert_map(svg_map_path)

    config = RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"my_map": map_definition}),
        sim_config=SimulationSettings(
            difficulty=difficulty, ped_density_by_difficulty=ped_densities
        ),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
    )
    env = make_robot_env(
        config=config,
        new_env=True,
        debug=True,
        recording_enabled=False,
    )

    return env


def run():
    env = make_env("maps/svg_maps/debug_05.svg")
    logger.error(f"Current env observation space: {env.observation_space}")

    model = PPO.load("./model/run_043", env=env)
    logger.info("Loading robot model from ./model/run_043")

    obs, _ = env.reset()
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        env.render()

        if done:
            obs, _ = env.reset()
            env.render()
    env.exit()


if __name__ == "__main__":
    SVG_MAP = "maps/svg_maps/debug_05.svg"
    run()
