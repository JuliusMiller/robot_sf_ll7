import loguru
import numpy as np
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from robot_sf.gym_env.env_config import PedEnvSettings
from robot_sf.gym_env.pedestrian_env import PedestrianEnv
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sim.sim_config import SimulationSettings

logger = loguru.logger


def training(svg_map_path: str):
    ped_densities = [0.01, 0.02, 0.04, 0.08]
    difficulty = 2
    rng = np.random.default_rng(42)

    # Create environment
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
        return PedestrianEnv(env_config, robot_model=robot_model)

    # Create vectorized environment
    env = make_vec_env(
        make_env,
        rng=rng,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    )

    # Create expert policy
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals/CartPole-v0",
        venv=env,
    )

    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(rollouts)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
    )

    reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print(f"Reward before training: {reward_before_training}")

    bc_trainer.train(n_epochs=1)
    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print(f"Reward after training: {reward_after_training}")

    # model.learn(total_timesteps=10_000_000, progress_bar=True, callback=combined_callback)
    # now = datetime.datetime.now()
    # filename = now.strftime("%Y-%m-%d_%H-%M-%S")
    # model.save(f"./model_ped/ppo_{filename}")
    # logger.info(f"Model saved as ppo_{filename}")


if __name__ == "__main__":
    SVG_MAP = "maps/svg_maps/debug_06.svg"

    training(SVG_MAP)
