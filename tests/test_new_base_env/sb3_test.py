import os

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from robot_sf.gym_env.robot_env_from_base import RobotEnvFromBase
from robot_sf.feature_extractor import DynamicsExtractor


def test_can_load_model_snapshot():
    MODEL_PATH = "./temp/ppo_model"
    MODEL_FILE = f"{MODEL_PATH}.zip"

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if os.path.exists(MODEL_FILE) and os.path.isfile(MODEL_FILE):
        os.remove(MODEL_FILE)

    n_envs = 2
    vec_env = make_vec_env(
        lambda: RobotEnvFromBase(), n_envs=n_envs, vec_env_cls=SubprocVecEnv
    )
    policy_kwargs = dict(features_extractor_class=DynamicsExtractor)
    model = PPO("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs)
    model.save(MODEL_PATH)
    assert os.path.exists(MODEL_FILE)

    inf_env = RobotEnvFromBase()
    model2 = PPO.load(MODEL_PATH, env=inf_env)

    obs, info = inf_env.reset()
    action, _ = model2.predict(obs, deterministic=True)

    assert action.shape == inf_env.action_space.shape

    if os.path.exists(MODEL_FILE) and os.path.isfile(MODEL_FILE):
        os.remove(MODEL_FILE)
        os.rmdir(os.path.dirname(MODEL_PATH))
