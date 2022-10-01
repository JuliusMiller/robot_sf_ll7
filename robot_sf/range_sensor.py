import math
# import json
# import os
import random
# from datetime import datetime
from dataclasses import dataclass
from typing import List

import numpy as np
# import matplotlib.pyplot as plt

from robot_sf.map import BinaryOccupancyGrid


@dataclass
class Range:
    lower: float
    upper: float


@dataclass
class LidarScannerSettings:
    lidar_range: int
    visualization_angle_portion: float
    lidar_n_rays: int

    # TODO: pre-compute the properties in __post_init__ constructor

    @property
    def scan_range(self) -> Range:
        return Range(0, self.lidar_range)

    @property
    def angle_opening(self) -> Range:
        return Range(-np.pi * self.visualization_angle_portion,
                      np.pi * self.visualization_angle_portion)

    @property
    def angle_increment(self) -> float:
        return (self.angle_opening.upper - self.angle_opening.lower) / (self.lidar_n_rays - 1)


class LiDARscanner():
    """The following class is responsible of creating a LiDAR scanner object"""

    def __init__(self, settings: LidarScannerSettings):
        scan_range = [settings.scan_range.lower, settings.scan_range.upper]
        angle_opening = [settings.angle_opening.lower, settings.angle_opening.upper]

        self.range = scan_range
        self.angle_opening = angle_opening
        self.angle_increment = settings.angle_increment
        self.distance_noise = 0
        self.angle_noise = 0
        self.num_readings = 0
        self.scan_structure = dict()
        self.scan_structure['properties'] = dict()
        self.scan_structure['data'] = dict()

        self.empty_scan_structure()

    def get_scan(self, x: float, y: float, orient: float,
                 input_map: BinaryOccupancyGrid, scan_noise: List[float]=None):
        """This method takes in input the state of the robot
        and an input map (map object) and returns a data structure
        containing the sensor readings"""
        self.empty_scan_structure()
        start_pt = [x, y]
        self.scan_structure['data']['pose'] = [x, y, orient]
        scan_length = self.num_readings

        if scan_noise is not None:
            lost_scans = np.where(np.random.random(scan_length) < scan_noise[0])[0]
            corrupt_scans = np.where(np.random.random(scan_length) < scan_noise[1])[0]
        else:
            lost_scans = []
            corrupt_scans = []

        for i in range(scan_length):
            #Transform orientation from laser frame to grid frame
            #and compute end-point
            grid_orient = orient + self.scan_structure['data']['angles'][i]
            #compute end-point
            end_pt = list()
            end_pt.append(start_pt[0] + self.range[1] * math.cos(grid_orient))
            end_pt.append(start_pt[1] + self.range[1] * math.sin(grid_orient))
            try:
                # TODO: find out why an exception might occur -> fix it and remove this try/catch block
                ray_index = input_map.raycast(np.array([start_pt]), np.array([end_pt]))
                flag, intercept_grid, intercept = input_map.does_ray_collide(ray_index)
                # added intercept grid for debugging purpose
            except Exception:
                flag = False

            if flag and i not in lost_scans:
                # ray collided with obstacle! Compute distance
                self.scan_structure['data']['ranges'][i] = \
                    math.sqrt((intercept[0] - start_pt[0])**2 + (intercept[1] - start_pt[1])**2)
                self.scan_structure['data']['cartesian'][i, 0] = intercept[0]
                self.scan_structure['data']['cartesian'][i, 1] = intercept[1]
                self.scan_structure['data']['grid'][i] = intercept_grid
                # added intercept grid for debugging purpose

            if scan_noise is not None and i in corrupt_scans and i not in lost_scans:
                scanned_distance = random.random() * self.range[1]
                self.scan_structure['data']['ranges'][i] = scanned_distance

                intercept = [0, 0]                
                intercept[0] = start_pt[0] + scanned_distance * math.cos(grid_orient)
                intercept[1] = start_pt[1] + scanned_distance * math.sin(grid_orient)

                self.scan_structure['data']['cartesian'][i, 0] = intercept[0]
                self.scan_structure['data']['cartesian'][i, 1] = intercept[1]
                self.scan_structure['data']['grid'][i] = input_map.convert_world_to_grid_no_error(np.array([intercept]))

        return self.scan_structure

    def empty_scan_structure(self):
        """This method empty the scan data structure and repopulate
        it, according to new properties if they have changed"""

        self._original_angles = np.arange(
            self.angle_opening[0],
            self.angle_opening[1] + self.angle_increment,
            self.angle_increment).tolist()
        self.num_readings = len(self._original_angles)

        self.scan_structure['properties']['range'] = self.range
        self.scan_structure['properties']['angle_opening'] = self.angle_opening
        self.scan_structure['properties']['angle_increment'] = self.angle_increment
        self.scan_structure['properties']['distance_noise'] = self.distance_noise
        self.scan_structure['data']['angles'] = self._original_angles
        self.scan_structure['data']['ranges'] = [math.nan] * self.num_readings
        self.scan_structure['data']['intensities'] = [math.inf] * self.num_readings
        self.scan_structure['data']['pose'] = [0] * 3 # TODO: WTF?
        self.scan_structure['data']['cartesian'] = np.zeros((self.num_readings, 2), dtype=float)
        # added intercept grid for debugging purpose
        self.scan_structure['data']['grid'] = np.zeros((self.num_readings, 2), dtype=int)

    # def apply_distance_noise(self):
    #     ranges = np.array(self.scan_structure['data']['ranges'], dtype=float)
    #     new_ranges = self.distance_noise * np.random.randn(ranges.shape[0]) + ranges
    #     self.scan_structure['data']['ranges'] = new_ranges.tolist()

    # def apply_angle_noise(self):
    #     angles = np.array(self.scan_structure['data']['angles'], dtype=float)
    #     new_angles = self.angle_noise * np.random.randn(angles.shape[0]) + angles
    #     self.scan_structure['data']['angles'] = new_angles.tolist()

    # def show(self, map_entry=None):
    #     '''This method plots the rays'''
    #     tmp = np.zeros((len(self.scan_structure['data']['angles']), 2), dtype=float)
    #     ranges_filtered = self.remove_invalid_data(self.scan_structure['data']['ranges'])
    #     tmp[:, 0] = self.scan_structure['data']['angles']
    #     tmp[:, 1] = ranges_filtered
    #     tmp[~np.isnan(tmp).any(axis=1)]

    #     if tmp.size == 0:
    #         #No rays to plot
    #         plt.plot(self.scan_structure['data']['pose'][0], self.scan_structure['data']['pose'][1])
    #         plt.text(self.scan_structure['data']['pose'][0], self.scan_structure['data']['pose'][1], 'G')
    #         plt.show()
    #     else:
    #         #define rotation_matrix
    #         pts_x = np.multiply(tmp[:,1],np.cos(tmp[:,0]))
    #         pts_y = np.multiply(tmp[:,1],np.sin(tmp[:,0]))

    #         theta = self.scan_structure['data']['pose'][2]
    #         R = np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta),math.cos(theta)]])

    #         #Transform data from scanner frame to world frame
    #         tmp = np.array([pts_x, pts_y])
    #         pts_transformed = np.dot(R, tmp)

    #         pts_transformed[0,:] += self.scan_structure['data']['pose'][0]
    #         pts_transformed[1,:] += self.scan_structure['data']['pose'][1]

    #         if map_entry:
    #             plt.imshow(map_entry.Occupancy,cmap = 'Greys',\
    #                 extent=[-map_entry.grid_origin[0], -map_entry.grid_origin[0] + map_entry.map_length, \
    #                 -map_entry.grid_origin[1], -map_entry.grid_origin[1] + map_entry.map_height])

    #         plt.plot(self.scan_structure['data']['pose'][0],self.scan_structure['data']['pose'][1],'o',color='black')
    #         plt.text(self.scan_structure['data']['pose'][0],self.scan_structure['data']['pose'][1],'G')
    #         pts = np.array(self.scan_structure['data']['cartesian'])
    #         plt.scatter(pts[:,0], pts[:,1], marker = '.', color = 'blue')
    #         plt.axis('equal')
    #         plt.show()

    # def save(self,filename):
    #     ''' This method will save the scan data structure into a json file
    #     specified by filename'''
    #     copied_scan = self.scan_structure
    #     copied_scan['info'] = dict()
    #     copied_scan['info']['author'] = 'Matteo Caruso'
    #     copied_scan['info']['email'] = 'matteo.caruso@phd.units.it'
    #     copied_scan['info']['created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    #     #Add path for saving json file
    #     filename, _ = os.path.splitext(filename)
    #     directory, name = os.path.split(filename)

    #     if not directory:
    #         #relative path
    #         home = os.getcwd()
    #         path = os.path.join(home, 'data', 'scans', f'{name}.json')
    #     else:
    #         #absolute path
    #         path = filename + '.json'

    #     with open(path,'w') as f:
    #         json.dump(self.scan_structure,f)
    #         print('Scan saved at:' + path)

    # def remove_invalid_data(self,range_list):
    #     '''This method removes invalid data from the ranges array,
    #     i.e. data which is below of range lower limit'''
    #     tmp = np.array(range_list, dtype=float)
    #     with np.errstate(invalid='ignore'):
    #         tmp[tmp < self.range[0]] = np.nan
    #     return tmp
