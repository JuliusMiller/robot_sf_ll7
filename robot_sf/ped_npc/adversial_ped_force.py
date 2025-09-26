from dataclasses import dataclass
from typing import Callable

import numba
import numpy as np
from pysocialforce.scene import PedState

from robot_sf.util.math import euclid_dist
from robot_sf.util.types import RobotPose, Vec2D


@dataclass
class AdversialPedForceConfig:
    is_active: bool = False
    relaxation_time: float = 0.5
    robot_radius: float = 1.0
    activation_threshold: float = 50.0
    force_multiplier: float = 3.0
    offset: float = 0.0
    target_ped_idx: int = -1


class AdversialPedForce:
    def __init__(
        self,
        config: AdversialPedForceConfig,
        peds: PedState,
        get_robot_pose: Callable[[], RobotPose],
    ):
        self.config = config
        self.peds = peds
        self.get_robot_pose = get_robot_pose
        self.last_forces = 0.0
        self.target_ped_idx = [0, -1]
        """Even if the target_idx restricts to one ped, groups forces may pull more pedestrians
            towards the robot"""

    def __call__(self) -> np.ndarray:
        threshold = (
            self.config.activation_threshold + self.peds.agent_radius + self.config.robot_radius
        )

        ped_positions = np.array(self.peds.pos(), dtype=np.float64)
        ped_velocities = np.array(self.peds.vel(), dtype=np.float64)
        ped_max_speeds = np.array(self.peds.max_speeds, dtype=np.float64)
        robot_pos = np.array(self.get_robot_pose()[0], dtype=np.float64)
        robot_orient = self.get_robot_pose()[1]
        forces = np.zeros((self.peds.size(), 2))

        adversial_ped_force(
            out_forces=forces,
            relaxation_time=self.config.relaxation_time,
            ped_positions=ped_positions,
            ped_velocities=ped_velocities,
            ped_max_speeds=ped_max_speeds,
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
    relaxation_time: float,
    ped_positions: np.ndarray,
    ped_velocities: np.ndarray,
    ped_max_speeds: np.ndarray,
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
    if isinstance(target_ped_idx, (int, np.integer)):
        indices = [target_ped_idx]
    else:
        indices = target_ped_idx

    for idx in indices:
        # Calculate attraction point in front of the robot
        attraction_point = np.empty(2)
        attraction_point[0] = robot_pos[0] + offset * np.cos(robot_orient)
        attraction_point[1] = robot_pos[1] + offset * np.sin(robot_orient)

        ped_pos = ped_positions[idx]
        distance = euclid_dist(attraction_point, ped_pos)

        desired = True

        if desired:
            if distance > 1e-6:  # avoid division by zero
                # Desired direction
                direction = (attraction_point - ped_pos) / distance

                # Desired velocity toward attraction point
                v_desired = direction * ped_max_speeds[idx]

                # Social force style: relaxation toward desired velocity
                out_forces[idx] = v_desired - ped_velocities[idx] / relaxation_time

        else:
            if distance < threshold:
                direction = attraction_point - ped_pos
                if distance > 1e-6:
                    out_forces[idx] = direction / distance * (threshold - distance)
