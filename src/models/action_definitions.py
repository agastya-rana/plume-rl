import enum
import functools
from typing import Callable

import numpy as np

from src.models.geometry import AngleField
from src.models.wind_directions import WindDirections


@enum.unique
class TurnActionEnum(enum.Enum):
    NO_TURN = enum.auto()
    UPWIND_TURN = enum.auto()
    CROSS_A_TURN = enum.auto()
    CROSS_B_TURN = enum.auto()
    DOWNWIND_TURN = enum.auto()


class TurnFunctions:
    """
    Instances require wind information. They have an attribute 'turn_functions' which is a dictionary
    mapping the enum items (no turn, upwind, etc.) to functions of fly angle. These functions take current fly angle
    and return the turn angle that must be added to current fly angle in order to satisfy the meaning of 'upwind turn,'
    'downwind turn,' etc
    """

    def __init__(self, wind_params: WindDirections):
        self.working_angular_deviation = AngleField()
        self.wind_params: WindDirections = wind_params
        self.turn_functions: dict[TurnActionEnum, Callable[[float], float]] = self.create_turn_functions()

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
    def no_turn(fly_orientation: float) -> float:
        return fly_orientation

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
