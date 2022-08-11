from abc import abstractmethod
from typing import Protocol

import numpy as np

PLUME_VIDEO_X_BOUNDS = np.array([0, 1500])  # From Nirag
PLUME_VIDEO_Y_BOUNDS = np.array([0, 900])  # From Nirag


class OdorPlume(Protocol):
    frame_number: int
    frame: np.ndarray

    def reset(self) -> list[int | np.ndarray]:
        pass

    def advance(self) -> list[int | np.ndarray]:
        pass


class OdorPlumeAllOnes:

    def __init__(self):
        self.frame_number: int = 0
        self.frame: np.ndarray = np.ones([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])

    def reset(self):
        self.__init__()
        return [self.frame_number, self.frame]

    def advance(self):
        self.frame_number += 1
        self.frame: np.ndarray = np.ones([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        return [self.frame_number, self.frame]


class OdorPlumeAllZeros:

    def __init__(self):
        self.frame_number: int = 0
        self.frame: np.ndarray = np.zeros([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])

    def reset(self):
        self.__init__()
        return [self.frame_number, self.frame]

    def advance(self):
        self.frame_number += 1
        self.frame: np.ndarray = np.zeros([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        return [self.frame_number, self.frame]


class OdorPlumeAlternating:
    def __init__(self):
        self.frame_number: int = 0
        self.frame: np.ndarray = np.ones([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])

    def reset(self):
        self.__init__()
        return [self.frame_number, self.frame]

    def advance(self):
        self.frame_number += 1
        if self.frame_number % 2 == 1:
            self.frame: np.ndarray = np.zeros([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        else:
            self.frame: np.ndarray = np.ones([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        return [self.frame_number, self.frame]
