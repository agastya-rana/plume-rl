from dataclasses import dataclass

import numpy as np

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
