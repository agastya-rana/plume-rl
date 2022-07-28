import numpy as np
from dataclasses import dataclass, field
from gym import Env, spaces
from gym.core import ActType, ObsType
from typing import Tuple, Optional, Union, Protocol, Callable

# Try to manage flow through "step" and "reset" with an event handler/observer pattern
from src.models.simulation_events import SimulationEvent, process_event

MAX_HISTORY_LENGTH = 1000
PLUME_VIDEO_Y_BOUNDS = np.array([0, 900])  # From Nirag
PLUME_VIDEO_X_BOUNDS = np.array([0, 1500])  # From Nirag


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
    value: np.ndarray = np.zeros(MAX_HISTORY_LENGTH)

    def update(self, sensor_location: np.ndarray, odor_plume: np.ndarray) -> np.ndarray:
        local_odor_concentration = detect_local_odor_concentration(sensor_location, odor_plume)
        updated_odor_history: np.ndarray = np.append(self.value, local_odor_concentration)
        updated_odor_history = np.delete(updated_odor_history, 0)
        self.value = updated_odor_history
        return self.value

    def clear(self):
        self.value = np.zeros(MAX_HISTORY_LENGTH)


class AngleField:

    def __get__(self, instance, owner) -> float:
        return instance.__dict__[self.name]

    def __set__(self, instance, value: float):
        instance.__dict__[self.name] = standardize_angle(value)

    def __set_name__(self, owner, name):
        self.name = name


@dataclass
class FlySpatialParameters:
    """
    Class that groups angle (in [0, 2pi)) and position information about a fly,
    and methods to update this information when the fly turns and walks
    """

    orientation: float = AngleField()
    position: np.ndarray = np.array([0, 0])
    randomization_x_bounds: np.ndarray = PLUME_VIDEO_X_BOUNDS
    randomization_y_bounds: np.ndarray = PLUME_VIDEO_Y_BOUNDS

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


class OdorPlume:

    def __init__(self):
        self.frame_number: int = 0
        self.frame: np.ndarray = np.ones([100, 100])

    def reset(self):
        self.__init__()

    def advance(self):
        self.frame_number += 1
        if self.frame_number % 2 == 0:
            self.frame = np.ones([100, 100])
        else:
            self.frame = np.zeros([100, 100])


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


class PlumeNavigationEnvironment(Env):

    def __init__(self, fly_spatial_parameters: FlySpatialParameters, odor_history: OdorHistory, odor_plume: OdorPlume):
        self.observation_space = spaces.Discrete(MAX_HISTORY_LENGTH)  # it's not obvious that this will be used
        self.observation = odor_history.value  # Test updates when the associated unique odor history object updates
        self.plume_x_bounds: np.ndarray = PLUME_VIDEO_X_BOUNDS  # Bounds taken from Nirag
        self.plume_y_bounds = np.ndarray = PLUME_VIDEO_Y_BOUNDS  # Bounds taken from Nirag

        # At the moment, it makes sense to construct this dictionary here
        # That way, everything related to organizing the simulation to match the gym protocol is in one place
        # In a sense, this dictionary *defines* the gym protocol of "Reset," "Step," etc
        self.subscribers: dict[SimulationEvent, list[Callable]] = {
            SimulationEvent.RESET: [fly_spatial_parameters.randomize_position,
                                    fly_spatial_parameters.randomize_orientation,
                                    odor_plume.reset,
                                    odor_history.clear]}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        rng: np.random.Generator = np.random.default_rng(seed)

        process_event(subscribers=self.subscribers, event_type=SimulationEvent.RESET)

        # odor_plume_frame = np.ones([self.plume_x_bounds[1], self.plume_y_bounds[1]])
        # self.odor_history.update(sensor_location=self.fly_params.position, odor_plume=odor_plume_frame)
        return self.observation

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        pass

    def render(self, mode="human"):
        pass
