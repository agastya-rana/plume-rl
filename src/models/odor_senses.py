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

    if np.sum(np.amax(speeds) == speeds) != 1:  # If there are ties for max speed, it's ambiguous and set to 0
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


def detect_local_odor_gradient_2D(fly_location: np.ndarray, odor_frame: np.ndarray):

    x, y = np.floor(fly_location).astype(int)
    half_antenna = np.floor(ANTENNA_LENGTH / 2)
    ys = np.arange(y - half_antenna, y + half_antenna).astype(int)
    xs = np.arange(x - half_antenna, x + half_antenna).astype(int)
    try:
        odor_array_y = odor_frame[x, ys]
        odor_array_y = odor_array_y.flatten()
        local_odor_gradient_y = linregress(ys, odor_array_y)[0]
    except IndexError:
        local_odor_gradient_y: float = 0
    try:
        odor_array_x = odor_frame[xs, y]
        odor_array_x = odor_array_x.flatten()
        local_odor_gradient_x = linregress(xs, odor_array_x)[0]
    except IndexError:
        local_odor_gradient_x: float = 0
    
    local_odor_gradient_arr = np.array([local_odor_gradient_x, local_odor_gradient_y])
    
    return local_odor_gradient_arr


def detect_local_odor_motion_2D(fly_location: np.ndarray, odor_frame_1: np.ndarray, odor_frame_2: np.ndarray):

    x, y = np.floor(fly_location).astype(int)
    half_antenna = np.floor(ANTENNA_LENGTH / 2)
    ys = np.arange(y - half_antenna, y + half_antenna).astype(int)
    xs = np.arange(x - half_antenna, x + half_antenna).astype(int)
    # print("in detect_local_odor_motion, this is odor_frame_1", odor_frame_1)
    # print("in detect_local_odor_motion, this is odor_frame_2", odor_frame_2)
    # print("this is the shape of odor_frame_1", odor_frame_1.shape)
    try:
        first_array_y = odor_frame_1[x, ys]
        second_array_y = odor_frame_2[x, ys]
        local_odor_motion_y = measure_odor_speed(odor_array_0=first_array_y, odor_array_1=second_array_y)
    except IndexError:
        local_odor_motion_y: float = 0
    try:
        first_array_x = odor_frame_1[xs, y]
        second_array_x = odor_frame_2[xs, y]
        local_odor_motion_x = measure_odor_speed(odor_array_0=first_array_x, odor_array_1=second_array_x)
    except IndexError:
        local_odor_motion_x: float = 0
    
    local_odor_motion = np.array([local_odor_motion_x, local_odor_motion_y])
    
    return local_odor_motion


def thresh_quantity(val):

    if val == 0:

        return 0

    elif val > 0:

        return 1

    else:

        return 2


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

@dataclass
class OdorFeatures_no_gradient:

    concentration: float = 0
    motion_speed: float = 0

    def update(self, sensor_location: np.ndarray, odor_plume_frame: np.ndarray, prior_odor_plume_frame: np.ndarray):
        self.concentration = detect_local_odor_concentration(fly_location=sensor_location,
                                                             odor_plume=odor_plume_frame)
        self.motion_speed = detect_local_odor_motion(fly_location=sensor_location,
                                                     odor_frame_1=prior_odor_plume_frame,
                                                     odor_frame_2=odor_plume_frame)

        return self.motion_speed, self.concentration

    def clear(self):
        self.concentration = 0
        self.motion_speed = 0
        return self.motion_speed, self.concentration

    def discretize_features(self):  # This should be factored somewhere else
        concentration = int(self.concentration > CONCENTRATION_THRESHOLD)

        if self.motion_speed == 0:
            motion_speed = 0
        elif self.motion_speed > 0:
            motion_speed = 1
        else:
            motion_speed = 2

        return np.array([concentration, motion_speed])

@dataclass
class OdorFeatures_no_motion:
    concentration: float = 0
    gradient: float = 0

    def update(self, sensor_location: np.ndarray, odor_plume_frame: np.ndarray, prior_odor_plume_frame: np.ndarray):
        self.concentration = detect_local_odor_concentration(fly_location=sensor_location,
                                                             odor_plume=odor_plume_frame)

        self.gradient = detect_local_odor_gradient(fly_location=sensor_location,
                                                   odor_frame=odor_plume_frame)

        return self.gradient, self.concentration

    def clear(self):
        self.concentration = 0
        self.gradient = 0
        return self.gradient, self.concentration

    def discretize_features(self):  # This should be factored somewhere else
        concentration = int(self.concentration > CONCENTRATION_THRESHOLD)
        if self.gradient == 0:
            gradient = 0
        elif self.gradient > 0:
            gradient = 1
        else:
            gradient = 2

        return np.array([concentration, gradient])


@dataclass 
class OdorFeatures_2D:

    def update(self, sensor_location: np.ndarray, odor_plume_frame: np.ndarray, prior_odor_plume_frame: np.ndarray):

        self.concentration = detect_local_odor_concentration(fly_location=sensor_location, odor_plume=odor_plume_frame)
        self.gradient = detect_local_odor_gradient_2D(sensor_location, odor_plume_frame)
        self.motion_speed = detect_local_odor_motion_2D(sensor_location, odor_frame_1 = prior_odor_plume_frame, odor_frame_2 = odor_plume_frame)

        return self.motion_speed, self.gradient, self.concentration


    def clear(self):

        self.concentration = 0
        self.gradient = np.zeros(2)
        self.motion_speed = np.zeros(2)

        return self.motion_speed, self.gradient, self.concentration


    def discretize_features(self):

        concentration = int(self.concentration > CONCENTRATION_THRESHOLD)


        first_grad = thresh_quantity(self.gradient[0])
        second_grad = thresh_quantity(self.gradient[1])

        first_mot = thresh_quantity(self.motion[0])
        second_mot = thresh_quantity(self.motion[1])

        return np.array([concentration, first_grad, second_grad, first_mot, second_mot])


class OdorFeatures_grad_2D:

    def update(self, sensor_location: np.ndarray, odor_plume_frame: np.ndarray, prior_odor_plume_frame: np.ndarray):

        self.concentration = detect_local_odor_concentration(fly_location=sensor_location, odor_plume=odor_plume_frame)
        self.gradient = detect_local_odor_gradient_2D(sensor_location, odor_plume_frame)
        self.motion_speed = detect_local_odor_motion(sensor_location, odor_frame_1 = prior_odor_plume_frame, odor_frame_2 = odor_plume_frame)

        return self.motion_speed, self.gradient, self.concentration


    def clear(self):

        self.concentration = 0
        self.gradient = np.zeros(2)
        self.motion_speed = 0

        return self.motion_speed, self.gradient, self.concentration


    def discretize_features(self):

        concentration = int(self.concentration > CONCENTRATION_THRESHOLD)


        first_grad = thresh_quantity(self.gradient[0])
        second_grad = thresh_quantity(self.gradient[1])

        mot = thresh_quantity(self.motion)

        return np.array([concentration, first_grad, second_grad, mot])



class OdorFeatures_motion_2D:

    def update(self, sensor_location: np.ndarray, odor_plume_frame: np.ndarray, prior_odor_plume_frame: np.ndarray):

        self.concentration = detect_local_odor_concentration(fly_location=sensor_location, odor_plume=odor_plume_frame)
        self.gradient = detect_local_odor_gradient(sensor_location, odor_plume_frame)
        self.motion_speed = detect_local_odor_motion_2D(sensor_location, odor_frame_1 = prior_odor_plume_frame, odor_frame_2 = odor_plume_frame)

        return self.motion_speed, self.gradient, self.concentration


    def clear(self):

        self.concentration = 0
        self.gradient = 0
        self.motion_speed = np.zeros(2)

        return self.motion_speed, self.gradient, self.concentration


    def discretize_features(self):

        concentration = int(self.concentration > CONCENTRATION_THRESHOLD)


        grad = thresh_quantity(self.gradient)

        first_mot = thresh_quantity(self.motion[0])
        second_mot = thresh_quantity(self.motion[1])

        return np.array([concentration, grad, first_mot, second_mot])



