from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress

CONCENTRATION_THRESHOLD = 100

ANTENNA_LENGTH = 14  # should be even

MAX_HISTORY_LENGTH = 1000


def detect_local_odor_concentration(fly_location: np.ndarray, odor_plume: np.ndarray) -> float:
    x, y = np.floor(fly_location).astype(int)
    try:
        local_odor_concentration: float = odor_plume[x, y]
    except IndexError:
        local_odor_concentration: float = 0
    return local_odor_concentration


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


def measure_odor_speed(odor_array_0: np.ndarray, odor_array_1: np.ndarray) -> int:
    odor_array_0_norm = odor_array_0 - np.mean(odor_array_0)
    odor_array_1_norm = odor_array_1 - np.mean(odor_array_1)
    delta_xs = np.arange(-1 * np.fix(ANTENNA_LENGTH / 2), 1 + np.fix(ANTENNA_LENGTH / 2))
    speeds = np.zeros(shape=delta_xs.shape)
    for index, delta_xi in enumerate(delta_xs):
        speeds[index] = np.mean([odor_array_0_norm[antenna_pos] * odor_array_1_norm[int(antenna_pos + delta_xi)]
                                 for antenna_pos in range(ANTENNA_LENGTH)
                                 if antenna_pos + delta_xi in range(ANTENNA_LENGTH)])


    if np.sum(np.amax(speeds) == speeds) != 1:# If there are ties for max speed, it's ambiguous and set to 0
        delta_x_hat = 0
    else:
        delta_x_hat = delta_xs[np.argmax(speeds)]
    return delta_x_hat


def detect_local_odor_motion(fly_location: np.ndarray, odor_frame_1: np.ndarray, odor_frame_2: np.ndarray) -> float:
    x, y = np.floor(fly_location).astype(int)
    half_antenna = np.floor(ANTENNA_LENGTH / 2)
    ys = np.arange(y - half_antenna, y + half_antenna).astype(int)
    try:
        first_array = odor_frame_1[x, ys]
        second_array = odor_frame_2[x, ys]
        local_odor_motion = measure_odor_speed(odor_array_0=first_array, odor_array_1=second_array)
    except IndexError:
        local_odor_motion: float = 0
    return local_odor_motion


def detect_local_odor_gradient(fly_location: np.ndarray, odor_frame: np.ndarray) -> float:
    x, y = np.floor(fly_location).astype(int)
    half_antenna = np.floor(ANTENNA_LENGTH / 2)
    ys = np.arange(y - half_antenna, y + half_antenna).astype(int)
    try:
        odor_array = odor_frame[x, ys]
        odor_array = odor_array.flatten()
        local_odor_gradient = linregress(ys, odor_array)[0]
    except IndexError:
        local_odor_gradient: float = 0
    return local_odor_gradient



@dataclass
class OdorFeatures:
    concentration: float = 0
    motion_speed: float = 0
    gradient: float = 0

    def update(self, sensor_location: np.ndarray, odor_plume_frame: np.ndarray, prior_odor_plume_frame: np.ndarray):
        self.concentration = detect_local_odor_concentration(fly_location=sensor_location,
                                                             odor_plume=odor_plume_frame)
        self.motion_speed = detect_local_odor_motion(fly_location=sensor_location,
                                                     odor_frame_1=prior_odor_plume_frame,
                                                     odor_frame_2=odor_plume_frame)
        self.gradient = detect_local_odor_gradient(fly_location=sensor_location,
                                                   odor_frame=odor_plume_frame)

        return self.motion_speed, self.gradient, self.concentration

    def clear(self):
        self.concentration = 0
        self.motion_speed = 0
        self.gradient = 0
        return self.motion_speed, self.gradient, self.concentration

    def discretize_features(self):  # This should be factored somewhere else
        concentration = int(self.concentration > CONCENTRATION_THRESHOLD)
        if self.gradient == 0:
            gradient = 0
        elif self.gradient > 0:
            gradient = 1
        else:
            gradient = 2

        if self.motion_speed == 0:
            motion_speed = 0
        elif self.motion_speed > 0:
            motion_speed = 1
        else:
            motion_speed = 2

        return np.array([concentration, gradient, motion_speed])
