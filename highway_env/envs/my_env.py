from math import floor, ceil
import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.highway_env import MOHighwayEnv
from highway_env.envs.common.action import Action
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

class MyEnv(MOHighwayEnv):
    """
    Multi-objective version of HighwayEnv
    """

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        # Create controlled vehicle
        self.controlled_vehicles = []
        vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
        vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
        self.controlled_vehicles.append(vehicle)
        self.road.vehicles.append(vehicle)

        # Create other vehicles
        for _ in range(ceil(self.config["vehicles_count"] * 0.5)):
            self._add_vehicle_front()
        for _ in range(floor(self.config["vehicles_count"] * 0.5)):
            self._add_vehicle_behind()
    
    def _add_vehicle_front(self) -> None:
        """Add vehicle in front of leading car"""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"], front=True)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

    def _add_vehicle_behind(self) -> None:
        """Add vehicle behind last car"""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"], front=False)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

    def _rewards(self, action: Action) -> dict:
        rewards = {}
        
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        rewards["speed"] = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])

        lanes = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        rewards["right"] = lane / max(len(lanes) - 1, 1)

        rewards["nocrash"] = 0 if self.vehicle.crashed else 1

        # acc_max = self.vehicle.ACC_MAX
        # front_vehicle, rear_vehicle = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.target_lane_index)
        # acc = self.vehicle.acceleration(ego_vehicle=self.vehicle, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)
        # norm_accel = utils.lmap(acc, [0,acc_max], [0,1])
        # rewards["acc"] = 1-norm_accel

        rewards["dist"] = np.min(
                [   
                    np.linalg.norm(v.position - self.vehicle.position) 
                    for v in self.road.vehicles 
                    if v is not self.vehicle
                ]
            )

        return rewards

    def _reward(self, action: Action) -> float:
        rewards = self._rewards(action)

        reward = rewards["speed"]
        return reward

register(
    id='my-env-v0',
    entry_point='highway_env.envs:MyEnv',
)