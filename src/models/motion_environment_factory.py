from enum import Enum
from typing import Type, Union

import numpy as np
from gym.spaces import Discrete

from src.models.action_definitions import WalkStopActionEnum, WalkStopDisplacements, DisplacementActionClass, \
    WalkDisplacements
from src.models.fly_spatial_parameters import FlySpatialParameters
from src.models.gym_motion_environment_classes import PlumeMotionNavigationEnvironment, \
    PlumeMotionPathIntegrationNavigationEnvironment
from src.models.integrator_senses import IntegratorSensor
from src.models.odor_senses import OdorHistory, OdorFeatures
from src.models.odor_plumes import OdorPlumeAllOnes, OdorPlumeAllZeros, OdorPlumeAlternating, OdorPlumeRollingRandom, \
    OdorPlumeFromMovie, MOVIE_PATH_1
from src.models.reward_schemes import RewardSchemeEnum
from src.models.wind_directions import WindDirections
from src.models.goals import GOAL_RADIUS


class WalkActionEnum:
    pass


class PlumeMotionNavigationBaseEnvironmentFactory:
    """
    The specific factories inherit from this class, which contains info required for any of them.
    This inheritance structure helped to get pytest to work, but it is cumbersome and should be changed...
    Note that the action enum should be injected into the environmet here, but it's not structured that way yet.
    """

    def __init__(self, movie_file_path=None,
                 action_enum: Union[Type[WalkActionEnum], Type[WalkStopActionEnum]] = WalkActionEnum,
                 action_class: DisplacementActionClass = WalkDisplacements):
        self.fly_spatial_parameters = FlySpatialParameters(orientation=0, position=np.array([0, 0]))
        self.odor_features = OdorFeatures()
        self.odor_plume_all_ones = OdorPlumeAllOnes()
        self.odor_plume_all_zeros = OdorPlumeAllZeros()
        self.odor_plume_alternating = OdorPlumeAlternating()
        self.odor_plume_rolling_random = OdorPlumeRollingRandom()
        if movie_file_path is None:
            self.odor_plume_movie1 = None
        else:
            self.odor_plume_movie1 = OdorPlumeFromMovie(movie_file_path=movie_file_path)
        self.wind_towards_east = WindDirections()
        self.wind_towards_south = WindDirections(wind_angle=-np.pi / 2)
        self.wind_towards_west = WindDirections(wind_angle=np.pi)
        self.wind_towards_north = WindDirections(wind_angle=np.pi / 2)
        self.path_integrator = IntegratorSensor(home_angle=0)
        self.action_enum = action_enum
        self.action_class = action_class  # default runs original design


class PlumeMotionNavigationEnvironmentMoviePlume1NiragRewardFactory(
    PlumeMotionNavigationBaseEnvironmentFactory):
    """
    This class has a property (plume environment) that is an instantiation of a parameterized motion plume
    environment. The plume here uses a plume movie from Nirag, and rewards whenever odor is above threshold.
    """

    def __init__(self, movie_file_path=MOVIE_PATH_1):
        super(PlumeMotionNavigationEnvironmentMoviePlume1NiragRewardFactory, self).__init__(
            movie_file_path=movie_file_path)

    @property
    def plume_environment(self) -> PlumeMotionNavigationEnvironment:
        return PlumeMotionNavigationEnvironment(wind_directions=self.wind_towards_east,
                                                fly_spatial_parameters=self.fly_spatial_parameters,
                                                odor_features=self.odor_features,
                                                odor_plume=self.odor_plume_movie1,
                                                reward_flag=RewardSchemeEnum.NIRAG_REWARD)


class PlumeMotionNavigationEnvironmentMovie1PlumeSourceRewardFactory(
    PlumeMotionNavigationBaseEnvironmentFactory):
    """
    This class has a property (plume environment) that is an instantiation of a parameterized motion plume
    environment. The plume here uses a movie from Nirag, and rewards when the agent gets to the source.
    """

    def __init__(self, movie_file_path=MOVIE_PATH_1):
        super(PlumeMotionNavigationEnvironmentMovie1PlumeSourceRewardFactory, self).__init__(
            movie_file_path=movie_file_path)

    @property
    def plume_environment(self) -> PlumeMotionNavigationEnvironment:
        return PlumeMotionNavigationEnvironment(wind_directions=self.wind_towards_east,
                                                fly_spatial_parameters=self.fly_spatial_parameters,
                                                odor_features=self.odor_features,
                                                odor_plume=self.odor_plume_movie1,
                                                reward_flag=RewardSchemeEnum.GOAL_ZONE,
                                                source_radius=GOAL_RADIUS)


class PlumeMotionNavigationEnvironmentPlumeAllOnesSimpleOdorHistRewardWestToEastWindFactory(
    PlumeMotionNavigationBaseEnvironmentFactory):
    """
    This class simply has a property (plume environment) that is an instantiation of a parameterized motion plume
    environment. The plume is always 1 in the arena. Used for debugging and testing.
    """

    @property
    def plume_environment(self) -> PlumeMotionNavigationEnvironment:
        return PlumeMotionNavigationEnvironment(wind_directions=self.wind_towards_east,
                                                fly_spatial_parameters=self.fly_spatial_parameters,
                                                odor_features=self.odor_features,
                                                odor_plume=self.odor_plume_all_ones,
                                                reward_flag=RewardSchemeEnum.SIMPLE_ODOR_HISTORY)


class PlumeMotionNavigationEnvironmentPlumeRollingSimpleOdorHistRewardWestToEastWindFactory(
    PlumeMotionNavigationBaseEnvironmentFactory):
    """
    This class simply has a property (plume environment) that is an instantiation of a parameterized motion plume
    environment. The plume is random values moving in the y-axis. Useful for testing.
    """

    @property
    def plume_environment(self) -> PlumeMotionNavigationEnvironment:
        return PlumeMotionNavigationEnvironment(wind_directions=self.wind_towards_east,
                                                fly_spatial_parameters=self.fly_spatial_parameters,
                                                odor_features=self.odor_features,
                                                odor_plume=self.odor_plume_rolling_random,
                                                reward_flag=RewardSchemeEnum.SIMPLE_ODOR_HISTORY)


class PlumeMotionNavigationEnvironmentPlumeRollingYMaxRewardWestToEastWindFactory(
    PlumeMotionNavigationBaseEnvironmentFactory):
    """
    This class simply has a property (plume environment) that is an instantiation of a parameterized motion plume
    environment.
    """

    @property
    def plume_environment(self) -> PlumeMotionNavigationEnvironment:
        return PlumeMotionNavigationEnvironment(wind_directions=self.wind_towards_east,
                                                fly_spatial_parameters=self.fly_spatial_parameters,
                                                odor_features=self.odor_features,
                                                odor_plume=self.odor_plume_rolling_random,
                                                reward_flag=RewardSchemeEnum.Y_MAX)


class PlumeMotionNavigationEnvironmentMovie1PlumeSourceRewardPathIntegratorFactory(
    PlumeMotionNavigationBaseEnvironmentFactory):
    """
    This class has a property (plume environment) that is an instantiation of a parameterized motion plume
    environment. The plume here uses a movie from Nirag, and rewards when the agent gets to the source.
    """

    def __init__(self, movie_file_path=MOVIE_PATH_1):
        super(PlumeMotionNavigationEnvironmentMovie1PlumeSourceRewardPathIntegratorFactory, self).__init__(
            movie_file_path=movie_file_path)

    @property
    def plume_environment(self) -> PlumeMotionPathIntegrationNavigationEnvironment:
        return PlumeMotionPathIntegrationNavigationEnvironment(wind_directions=self.wind_towards_east,
                                                               fly_spatial_parameters=self.fly_spatial_parameters,
                                                               odor_features=self.odor_features,
                                                               path_integrator=self.path_integrator,
                                                               odor_plume=self.odor_plume_movie1,
                                                               reward_flag=RewardSchemeEnum.GOAL_ZONE,
                                                               source_radius=GOAL_RADIUS)


class PlumeMotionNavigationEnvironmentMovie1PlumeSourceRewardStopActionFactory(
    PlumeMotionNavigationBaseEnvironmentFactory):
    """
    This class has a property (plume environment) that is an instantiation of a parameterized motion plume
    environment. The plume here uses a movie from Nirag, and rewards when the agent gets to the source.
    """

    def __init__(self, movie_file_path=MOVIE_PATH_1, actions=WalkStopActionEnum, action_class=WalkStopDisplacements):
        super(PlumeMotionNavigationEnvironmentMovie1PlumeSourceRewardStopActionFactory, self).__init__(
            movie_file_path=movie_file_path,
            action_enum=actions,
            action_class=WalkStopDisplacements)

    @property
    def plume_environment(self) -> PlumeMotionNavigationEnvironment:
        return PlumeMotionNavigationEnvironment(wind_directions=self.wind_towards_east,
                                                fly_spatial_parameters=self.fly_spatial_parameters,
                                                odor_features=self.odor_features,
                                                odor_plume=self.odor_plume_movie1,
                                                reward_flag=RewardSchemeEnum.GOAL_ZONE,
                                                source_radius=GOAL_RADIUS,
                                                action_enum=self.action_enum,
                                                action_class=self.action_class
                                                )
