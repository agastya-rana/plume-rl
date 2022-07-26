import numpy as np
from dataclasses import dataclass, field

MAX_HISTORY_LENGTH = 1000


def standardize_angle(angle: float) -> float:
    angle = angle % (2 * np.pi)
    return angle


def angle_to_unit_vector(angle: float) -> np.ndarray:
    unit_vector: np.ndarray = np.array([np.cos(angle), np.sin(angle)])
    return unit_vector


def detect_local_odor_concentration(fly_location: np.ndarray, odor_plume: np.ndarray) -> float:
    x, y = np.floor(fly_location).astype(int)
    local_odor_concentration: float = odor_plume[x, y]
    return local_odor_concentration


@dataclass
class OdorHistory:
    odor_history: np.ndarray = np.zeros(MAX_HISTORY_LENGTH)

    def update_odor_history(self, sensor_location: np.ndarray, odor_plume: np.ndarray) -> np.ndarray:
        local_odor_concentration = detect_local_odor_concentration(sensor_location, odor_plume)
        updated_odor_history: np.ndarray = np.append(self.odor_history, local_odor_concentration)
        updated_odor_history = np.delete(updated_odor_history, 0)
        self.odor_history = updated_odor_history
        return self.odor_history


class AngleField:

    def __get__(self, instance, owner) -> float:
        return instance.__dict__[self.name]

    def __set__(self, instance, value: float):
        instance.__dict__[self.name] = standardize_angle(value)

    def __set_name__(self, owner, name):
        self.name = name


# Note tried to implement a standardized angle descriptor but too many arithmetic dunder methods to implement

@dataclass
class FlySpatialParameters:
    """
    Class that groups angle (in [0, 2pi)) and position information about a fly,
    and methods to update this information when the fly turns and walks
    """

    orientation: float = AngleField()
    position: np.ndarray = np.array([0, 0])

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
