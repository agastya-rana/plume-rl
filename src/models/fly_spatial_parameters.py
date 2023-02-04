from dataclasses import dataclass

import numpy as np

from src.models.geometry import AngleField, angle_to_unit_vector

X_RANDOMIZATION_BOUNDS = np.array([1000, 1200])
Y_RANDOMIZATION_BOUNDS = np.array([310, 590])


@dataclass
class FlySpatialParameters:
    """
    Class that groups angle (in [0, 2pi)) and position information about a fly,
    and methods to update this information when the fly turns and walks
    """

    orientation: float = AngleField()
    position: np.ndarray = np.array([0, 0])
    integrator_origin: np.ndarray = np.array([0, 0])
    home_vector: np.ndarray = np.array([0, 0])

    def update_position(self, walking_direction: np.ndarray) -> np.ndarray:
        self.position = self.position + walking_direction
        self.home_vector = self.integrator_origin - self.position
        return self.position

    def reset_integrator(self):
        self.integrator_origin = self.position
        self.home_vector = np.array([0, 0])

    def turn(self, turn_angle: float) -> float:
        self.orientation += turn_angle
        return self.orientation

    def turn_and_walk(self, turn_angle: float) -> np.ndarray:
        self.turn(turn_angle)
        walking_direction: np.ndarray = angle_to_unit_vector(self.orientation)
        self.update_position(walking_direction)
        return self.position

    def randomize_position(self, rng: np.random.Generator = np.random.default_rng(12345),
                           x_bounds: np.ndarray = X_RANDOMIZATION_BOUNDS,
                           y_bounds: np.ndarray = Y_RANDOMIZATION_BOUNDS,
                           valid_locations: np.ndarray = None) -> np.ndarray:
        if valid_locations is not None:
            x_too_high = valid_locations[:, 0] > x_bounds[1]
            x_too_low = valid_locations[:, 0] < x_bounds[0]
            y_too_high = valid_locations[:, 1] > y_bounds[1]
            y_too_low = valid_locations[:, 1] < y_bounds[0]
            keep_indices = ~x_too_low & ~x_too_high & ~y_too_low & ~y_too_high
            valid_locations = valid_locations[keep_indices, :]
            n_valid_locations = valid_locations.shape[0]
            assert n_valid_locations > 0
            random_valid_index = rng.choice(n_valid_locations, 1)
            random_x = valid_locations[random_valid_index, 0].item()
            random_y = valid_locations[random_valid_index, 1].item()

        else:
            random_x = rng.uniform(
                low=x_bounds[0],
                high=x_bounds[1])
            random_y = rng.uniform(
                low=y_bounds[0],
                high=y_bounds[1])
        random_position = np.array([random_x, random_y])
        self.position = random_position
        self.reset_integrator()
        return self.position

    def randomize_orientation(self, rng: np.random.Generator = np.random.default_rng(12345)) -> float:
        self.orientation = rng.uniform(
            low=0,
            high=2 * np.pi)
        return self.orientation

    def distance(self, distance_from: np.ndarray) -> bool:
        distance = np.sqrt(
            np.square(self.position[0] - distance_from[0]) + np.square(self.position[1] - distance_from[1]))
        return distance
