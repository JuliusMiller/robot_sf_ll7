import pytest
from robot_sf.sim_config import (
    EnvSettings,
    SimulationSettings,
    LidarScannerSettings,
    DifferentialDriveSettings,
    BicycleDriveSettings,
    MapDefinitionPool,
    DifferentialDriveRobot,
    BicycleDriveRobot)

def test_env_settings_initialization():
    env_settings = EnvSettings()
    assert isinstance(env_settings.sim_config, SimulationSettings)
    assert isinstance(env_settings.lidar_config, LidarScannerSettings)
    assert isinstance(env_settings.robot_config, DifferentialDriveSettings)
    assert isinstance(env_settings.map_pool, MapDefinitionPool)

def test_env_settings_post_init():
    with pytest.raises(ValueError):
        env_settings = EnvSettings(sim_config=None)

def test_robot_factory():
    env_settings = EnvSettings()
    robot = env_settings.robot_factory()
    assert isinstance(robot, DifferentialDriveRobot)

    env_settings.robot_config = BicycleDriveSettings()
    robot = env_settings.robot_factory()
    assert isinstance(robot, BicycleDriveRobot)

    with pytest.raises(NotImplementedError):
        env_settings.robot_config = "unsupported type"
        env_settings.robot_factory()
