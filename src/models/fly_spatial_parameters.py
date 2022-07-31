from dataclasses import dataclass

import numpy as np

from src.models.geometry import AngleField, angle_to_unit_vector

X_RANDOMIZATION_BOUNDS = np.array([10, 1490])
Y_RANDOMIZATION_BOUNDS = np.array([10, 890])


@dataclass
class FlySpatialParameters:
    """
    Class that groups angle (in [0, 2pi)) and position information about a fly,
    and methods to update this information when the fly turns and walks
    """

    orientation: float = AngleField()
    position: np.ndarray = np.array([0, 0])
    randomization_x_bounds: np.ndarray = X_RANDOMIZATION_BOUNDS
    randomization_y_bounds: np.ndarray = Y_RANDOMIZATION_BOUNDS

    def update_position(self, walking_direction: np.ndarray) -> np.ndarray:
        self.position = self.position + walking_direction
        return self.position

    def turn(self, turn_angle: float) -> float:
        self.orientation += turn_angle
        return self.orientation

    def turn_and_walk(self, turn_angle: float) -> np.ndarray:
        self.turn(turn_angle)
        walking_direction: np.ndarray = angle_to_unit_vector(self.orientation)
        self.update_position(walking_direction)
        return self.position

    def randomize_position(self, rng: np.random.Generator = np.random.default_rng(12345)) -> np.ndarray:
        random_x = rng.uniform(
            low=self.randomization_x_bounds[0],
            high=self.randomization_x_bounds[1])
        random_y = rng.uniform(
            low=self.randomization_y_bounds[0],
            high=self.randomization_y_bounds[1])
        random_position = np.array([random_x, random_y])
        self.position = random_position
        return self.position

    def randomize_orientation(self, rng: np.random.Generator = np.random.default_rng(12345)) -> float:
        self.orientation = rng.uniform(
            low=0,
            high=2 * np.pi)
        return self.orientation
