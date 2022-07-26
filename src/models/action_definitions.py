import enum
import functools
from typing import Callable

import numpy as np

from src.models.odor_navigation_environment import WindDirections, FlySpatialParameters, AngleField


@enum.unique
class TurnActionEnum(enum.Enum):
    NO_TURN = enum.auto()
    UPWIND_TURN = enum.auto()
    CROSS_A_TURN = enum.auto()
    CROSS_B_TURN = enum.auto()
    DOWNWIND_TURN = enum.auto()


class TurnFunctions:
    working_angular_deviation = AngleField()

    def __init__(self, wind_params: WindDirections):
        self.wind_params = wind_params
        self.turn_functions = self.create_turn_functions()

    def turn_against_orientation_sign(self, fly_orientation: float, orientation: float) -> int:
        """
        There is an orientation (like wind direction) to turn against (like an
        upwind turn). This function returns the sign of the ccw turn the fly must make
        to turn against the given orientation
        """
        self.working_angular_deviation = orientation - fly_orientation
        if self.working_angular_deviation > np.pi:  # the ccw rotation taking fly to wind is the short way
            sign = 1  # so turn ccw into the wind (turns should be <180 degrees)
        else:  # the ccw rotation taking fly to wind is the long way
            sign = -1  # so turn cw into the wind
        return sign

    def turn_against_orientation(self, fly_orientation: float, orientation: float) -> float:
        turn_magnitude = np.pi / 6
        turn_sign = self.turn_against_orientation_sign(fly_orientation=fly_orientation, orientation=orientation)
        return turn_sign * turn_magnitude

    @staticmethod
    def no_turn(current_orientation: float) -> float:
        return current_orientation

    def create_turn_functions(self) -> dict[TurnActionEnum, Callable[[float], float]]:
        turn_funcs = {TurnActionEnum.NO_TURN: self.no_turn,
                      TurnActionEnum.UPWIND_TURN: functools.partial(
                          self.turn_against_orientation, orientation=self.wind_params.wind_angle),
                      TurnActionEnum.CROSS_A_TURN: functools.partial(
                          self.turn_against_orientation, orientation=self.wind_params.crosswind_a),
                      TurnActionEnum.CROSS_B_TURN: functools.partial(
                          self.turn_against_orientation, orientation=self.wind_params.crosswind_b),
                      TurnActionEnum.DOWNWIND_TURN: functools.partial(
                          self.turn_against_orientation, orientation=self.wind_params.downwind)}
        return turn_funcs
