from typing import cast

import numpy as np


def standardize_angle(angle: float) -> float:
    angle = angle % (2 * np.pi)
    return angle


def angle_to_unit_vector(angle: float) -> np.ndarray:
    unit_vector: np.ndarray = np.array([np.cos(angle), np.sin(angle)])
    return unit_vector


class AngleField:

    def __get__(self, instance, owner) -> float:
        return cast(float, instance.__dict__[self.name])

    def __set__(self, instance, value: float):
        instance.__dict__[self.name] = standardize_angle(value)

    def __set_name__(self, owner, name):
        self.name = name
