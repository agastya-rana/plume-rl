from dataclasses import dataclass

import numpy as np

from src.models.geometry import angle_to_unit_vector, AngleField

GOAL_RADIUS = 100

GOAL_Y = 450

GOAL_X = 150

MAX_HISTORY_LENGTH = 1000
X_RANDOMIZATION_BOUNDS = np.array([10, 1490])
Y_RANDOMIZATION_BOUNDS = np.array([10, 890])


def detect_local_odor_concentration(fly_location: np.ndarray, odor_plume: np.ndarray) -> float:
    x, y = np.floor(fly_location).astype(int)
    try:
        local_odor_concentration: float = odor_plume[x, y]
    except IndexError:
        local_odor_concentration: float = 0
    return local_odor_concentration


class WindDirections:
    wind_angle: float = AngleField()
    crosswind_a: float = AngleField()
    crosswind_b: float = AngleField()
    downwind: float = AngleField()

    def __init__(self, wind_angle: float = 0):
        self.wind_angle = wind_angle
        self.crosswind_a = wind_angle + (np.pi / 2)
        self.crosswind_b = wind_angle - (np.pi / 2)
        self.downwind = wind_angle + np.pi

    def __repr__(self):
        return 'WindDirections(wind = {0})'.format(self.wind_angle)


@dataclass
class OdorHistory:
    value: np.ndarray = np.zeros(MAX_HISTORY_LENGTH)

    def update(self, sensor_location: np.ndarray, odor_plume_frame: np.ndarray) -> np.ndarray:
        local_odor_concentration = detect_local_odor_concentration(sensor_location, odor_plume_frame)
        updated_odor_history: np.ndarray = np.append(self.value, local_odor_concentration)
        updated_odor_history = np.delete(updated_odor_history, 0)
        self.value = updated_odor_history
        return self.value

    def clear(self):
        self.value = np.zeros(MAX_HISTORY_LENGTH)
        return self.value


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


@dataclass
class GoalZone:
    goal_x: int | float = GOAL_X
    goal_y: int | float = GOAL_Y
    goal_radius: int | float = GOAL_RADIUS

    def is_in_goal_zone(self, test_position: np.ndarray) -> bool:
        x_deviation = test_position[0] - self.goal_x
        y_deviation = test_position[1] - self.goal_y
        deviation = np.sqrt(x_deviation ** 2 + y_deviation ** 2)
        return deviation < self.goal_radius


