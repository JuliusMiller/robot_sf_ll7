from dataclasses import dataclass
from typing import Callable

import numba
import numpy as np
from pysocialforce.scene import PedState

from robot_sf.util.types import Vec2D


@dataclass
class AdversialPedForceConfig:
    is_active: bool = True
    robot_radius: float = 1.0
    activation_threshold: float = 50.0
    force_multiplier: float = 2.0


class AdversialPedForce:
    def __init__(
        self,
        config: AdversialPedForceConfig,
        peds: PedState,
        get_robot_pos: Callable[[], Vec2D],
        target_ped_idx: int = -1,
    ):
        self.config = config
        self.peds = peds
        self.get_robot_pos = get_robot_pos
        self.last_forces = 0.0
        self.target_ped_idx = target_ped_idx

    def __call__(self) -> np.ndarray:
        threshold = (
            self.config.activation_threshold + self.peds.agent_radius + self.config.robot_radius
        )
        ped_positions = np.array(self.peds.pos(), dtype=np.float64)
        robot_pos = np.array(self.get_robot_pos(), dtype=np.float64)
        forces = np.zeros((self.peds.size(), 2))
        adversial_ped_force(forces, ped_positions, robot_pos, threshold, self.target_ped_idx)
        forces = forces * self.config.force_multiplier
        self.last_forces = forces
        return forces


@numba.njit(fastmath=True)
def adversial_ped_force(
    out_forces: np.ndarray,
    ped_positions: np.ndarray,
    robot_pos: Vec2D,
    threshold: float,
    target_ped_idx: int,
):
    """
    Compute the attractive forces pulling each pedestrian towards the robot if within a threshold.

    Parameters
    ----------
    out_forces : np.ndarray
        Output array for computed forces (shape: [num_peds, 2]).
    ped_positions : np.ndarray
        Array of pedestrian positions (shape: [num_peds, 2]).
    robot_pos : Vec2D
        Position of the robot (length-2 array).
    threshold : float
        Only apply force if pedestrian is within this distance to the robot.

    Returns
    -------
    None
    """
    for i, ped_pos in enumerate(ped_positions):
        if i == target_ped_idx:
            distance = euclid_dist(robot_pos, ped_pos)
            if distance < threshold:
                direction = robot_pos - ped_pos
                norm = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
                if norm > 1e-6:
                    out_forces[i] = direction / norm * (threshold - distance)
                else:
                    out_forces[i] = np.zeros(2)
            else:
                out_forces[i] = np.zeros(2)
        else:
            out_forces[i] = np.zeros(2)


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
