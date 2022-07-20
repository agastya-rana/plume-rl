import numpy as np
import numpy.typing as npt
import gym
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class CenterStationaryFly:
    position: npt.NDArray = np.full(2, np.nan)
    speed: float = np.nan
    angle: float = np.nan

    def reset(self):
        self.position = np.zeros(2)
        self.speed = 0
        self.angle = 0

@dataclass
class AcceleratingRotatingFly:
    position: npt.NDArray = np.full(2, np.nan)
    speed: float = np.nan
    angle: float = np.nan

    def reset(self):
        self.position = np.zeros(2)
        self.speed = 0.01
        self.angle = 0

    def step(self):
        self.position = self.position + np.array([self.speed, 0])
        self.speed += 0.01
        self.angle += 0.5 * np.pi


class OdorPlume(ABC):
    @abstractmethod
    def reset(self):
        """Set initial conditions of plume"""

    @abstractmethod
    def step(self):
        """Advance plume one time step"""

class NoOdorPlume(OdorPlume):

    def __init__(self):
        self.snapshot = np.nan

    def reset(self):
        self.snapshot = 0

    def step(self):
        pass


class AlternatingOdorPlume:
    def __init__(self):
        self.snapshot = np.nan

    def reset(self):
        self.snapshot = 0

    def step(self):
        if self.snapshot == 0:
            self.snapshot = 1
        elif self.snapshot == 1:
            self.snapshot = 0
        else:
            raise ValueError('Unexpected snapshot value at step time: {0}'.format(self.snapshot))


class NoOdorCenterUpwindStationaryFlyOdorNavigationEnvironment(gym.Env):

    def __init__(self) -> None:
        self.fly = CenterStationaryFly()
        self.plume = NoOdorPlume()

    def reset(self, **kwargs) -> None:
        self.fly.reset()
        self.plume.reset()
