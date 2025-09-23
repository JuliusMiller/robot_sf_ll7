from dataclasses import dataclass
from typing import Callable

import numba
import numpy as np
from pysocialforce.scene import PedState

from robot_sf.util.types import RobotPose, Vec2D


@dataclass
class AdversialPedForceConfig:
    is_active: bool = True
    robot_radius: float = 1.0
    activation_threshold: float = 50.0
    force_multiplier: float = 2.0
    offset: float = 0.0


class AdversialPedForce:
    def __init__(
        self,
        config: AdversialPedForceConfig,
        peds: PedState,
        get_robot_pose: Callable[[], RobotPose],
        target_ped_idx: int = -1,
    ):
        self.config = config
        self.peds = peds
        self.get_robot_pose = get_robot_pose
        self.last_forces = 0.0
        self.target_ped_idx = target_ped_idx
        """Even if the target_idx restricts to one ped, groups forces may pull more pedestrians
            towards the robot"""

    def __call__(self) -> np.ndarray:
        threshold = (
            self.config.activation_threshold + self.peds.agent_radius + self.config.robot_radius
        )
        ped_positions = np.array(self.peds.pos(), dtype=np.float64)
        robot_pos = np.array(self.get_robot_pose()[0], dtype=np.float64)
        robot_orient = self.get_robot_pose()[1]
        forces = np.zeros((self.peds.size(), 2))
        adversial_ped_force(
            out_forces=forces,
            ped_positions=ped_positions,
            robot_pos=robot_pos,
            robot_orient=robot_orient,
            offset=self.config.offset,
            threshold=threshold,
            target_ped_idx=self.target_ped_idx,
        )
        forces = forces * self.config.force_multiplier
        self.last_forces = forces
        return forces


@numba.njit(fastmath=True)
def adversial_ped_force(
    out_forces: np.ndarray,
    ped_positions: np.ndarray,
    robot_pos: Vec2D,
    robot_orient: float,
    offset: float,
    threshold: float,
    target_ped_idx: int,
):
    """
    Compute the attractive force pulling the target pedestrian towards a point in front of the robot
      specified by the offset .

    Parameters
    ----------
    out_forces : np.ndarray
        Output array for computed forces (shape: [num_peds, 2]).
    ped_positions : np.ndarray
        Array of pedestrian positions (shape: [num_peds, 2]).
    robot_pos : Vec2D
        Position of the robot (length-2 array).
    robot_orient : float
        Orientation of the robot in radians.
    offset : float
        Distance in front of the robot to compute the attraction point.
    threshold : float
        Only apply force if pedestrian is within this distance to the attraction point.
    target_ped_idx : int
        Index of the pedestrian to apply the force to.

    Returns
    -------
    None
    """
    # Calculate attraction point in front of the robot
    attraction_point = np.empty(2)
    attraction_point[0] = robot_pos[0] + offset * np.cos(robot_orient)
    attraction_point[1] = robot_pos[1] + offset * np.sin(robot_orient)

    ped_pos = ped_positions[target_ped_idx]
    distance = euclid_dist(attraction_point, ped_pos)
    if distance < threshold:
        direction = attraction_point - ped_pos
        norm = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
        if norm > 1e-6:
            out_forces[target_ped_idx] = direction / norm * (threshold - distance)


# TODO: REFACTOR TO UTILS FILE -> euclid_dist is defined in range_sensor.py
@numba.njit(fastmath=True)
def euclid_dist(v_1: Vec2D, v_2: Vec2D) -> float:
    """
    Compute the Euclidean distance between two 2D vectors.

    This function uses the standard formula for Euclidean distance: sqrt((x1 - x2)^2 + (y1 - y2)^2).

    Parameters
    ----------
    v_1 : Vec2D
        The first 2D vector. This is a tuple or list of two numbers representing
        the x and y coordinates.
    v_2 : Vec2D
        The second 2D vector. This is a tuple or list of two numbers representing
        the x and y coordinates.

    Returns
    -------
    float
        The Euclidean distance between `v_1` and `v_2`.
    """
    # Compute the difference in x coordinates and square it
    x_diff_sq = (v_1[0] - v_2[0]) ** 2
    # Compute the difference in y coordinates and square it
    y_diff_sq = (v_1[1] - v_2[1]) ** 2
    # Return the square root of the sum of the squared differences
    return (x_diff_sq + y_diff_sq) ** 0.5
