import numpy as np

from src.models.fly_spatial_parameters import FlySpatialParameters
from src.models.gym_history_environment_classes import PlumeHistoryNavigationEnvironment
from src.models.odor_senses import OdorHistory
from src.models.odor_plumes import OdorPlumeAllOnes, OdorPlumeAllZeros, OdorPlumeAlternating
from src.models.reward_schemes import RewardSchemeEnum
from src.models.wind_directions import WindDirections


class PlumeHistoryNavigationBaseEnvironmentFactory:
    def __init__(self):
        self.fly_spatial_parameters = FlySpatialParameters(orientation=0, position=np.array([0, 0]))
        self.odor_history = OdorHistory()
        self.odor_plume_all_ones = OdorPlumeAllOnes()
        self.odor_plume_all_zeros = OdorPlumeAllZeros()
        self.odor_plume_alternating = OdorPlumeAlternating()
        self.westerly_wind = WindDirections()
        self.northerly_wind = WindDirections(wind_angle=-np.pi / 2)
        self.easterly_wind = WindDirections(wind_angle=np.pi)
        self.southerly_wind = WindDirections(wind_angle=np.pi / 2)


class PlumeNavigationEnvironmentPlumeAllOnesSimpleOdorHistRewardWesterlyWindFactory(
                                         PlumeHistoryNavigationBaseEnvironmentFactory):
    @property
    def plume_environment(self) -> PlumeHistoryNavigationEnvironment:
        return PlumeHistoryNavigationEnvironment(wind_directions=self.westerly_wind,
                                                 fly_spatial_parameters=self.fly_spatial_parameters,
                                                 odor_history=self.odor_history,
                                                 odor_plume=self.odor_plume_all_ones,
                                                 reward_flag=RewardSchemeEnum.SIMPLE_ODOR_HISTORY)


class PlumeNavigationEnvironmentPlumeAllOnesSimpleOdorHistRewardNortherlyWindFactory(
                                       PlumeHistoryNavigationBaseEnvironmentFactory):
    @property
    def plume_environment(self) -> PlumeHistoryNavigationEnvironment:
        return PlumeHistoryNavigationEnvironment(wind_directions=self.northerly_wind,
                                                 fly_spatial_parameters=self.fly_spatial_parameters,
                                                 odor_history=self.odor_history,
                                                 odor_plume=self.odor_plume_all_ones,
                                                 reward_flag=RewardSchemeEnum.SIMPLE_ODOR_HISTORY)


class PlumeNavigationEnvironmentAlternatingPlumeSimpleOdorHistRewardWesterlyWindFactory(
                                                        PlumeHistoryNavigationBaseEnvironmentFactory):
    @property
    def plume_environment(self) -> PlumeHistoryNavigationEnvironment:
        return PlumeHistoryNavigationEnvironment(wind_directions=self.westerly_wind,
                                                 fly_spatial_parameters=self.fly_spatial_parameters,
                                                 odor_history=self.odor_history,
                                                 odor_plume=self.odor_plume_alternating,
                                                 reward_flag=RewardSchemeEnum.SIMPLE_ODOR_HISTORY)


class PlumeNavigationEnvironmentPlumeZerosSimpleOdorHistRewardWesterlyWindFactory(
                                             PlumeHistoryNavigationBaseEnvironmentFactory):
    @property
    def plume_environment(self) -> PlumeHistoryNavigationEnvironment:
        return PlumeHistoryNavigationEnvironment(wind_directions=self.westerly_wind,
                                                 fly_spatial_parameters=self.fly_spatial_parameters,
                                                 odor_history=self.odor_history,
                                                 odor_plume=self.odor_plume_all_zeros,
                                                 reward_flag=RewardSchemeEnum.SIMPLE_ODOR_HISTORY)


class PlumeNavigationEnvironmentPlumeAlternatingGoalZoneRewardWesterlyWindFactory(
                                     PlumeHistoryNavigationBaseEnvironmentFactory):
    @property
    def plume_environment(self) -> PlumeHistoryNavigationEnvironment:
        return PlumeHistoryNavigationEnvironment(wind_directions=self.westerly_wind,
                                                 fly_spatial_parameters=self.fly_spatial_parameters,
                                                 odor_history=self.odor_history,
                                                 odor_plume=self.odor_plume_alternating,
                                                 reward_flag=RewardSchemeEnum.GOAL_ZONE)
