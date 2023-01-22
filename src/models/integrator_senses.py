from dataclasses import dataclass
import numpy as np

# TO DO: should home_distance have a discretize method?

ANGLE_BIN_NUMBER = 4


@dataclass
class IntegratorSensor:
    home_angle: float = 0
    home_distance: float = 0

    def update(self, homing_vector: np.ndarray):
        if np.array_equal(np.array([0, 0]), homing_vector):
            self.home_angle = 0
        else:
            self.home_angle = np.arctan2(homing_vector[1], homing_vector[0])
        self.home_distance = np.linalg.norm(homing_vector)

    def clear(self):
        self.home_angle = 0
        self.home_distance = 0

    def discretize_angle(self, angle_bin_number: int = ANGLE_BIN_NUMBER) -> int:
        angle_bins = np.linspace(0, np.pi, angle_bin_number)
        return np.digitize(self.home_angle, angle_bins).item()
