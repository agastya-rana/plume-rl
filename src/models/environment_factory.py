import numpy as np

from src.models.fly_spatial_parameters import FlySpatialParameters
from src.models.gym_environment_classes import PlumeNavigationEnvironment
from src.models.odor_histories import OdorHistory
from src.models.odor_plumes import OdorPlumeAllOnes, OdorPlumeAllZeros, OdorPlumeAlternating
from src.models.reward_schemes import RewardSchemeEnum
from src.models.wind_directions import WindDirections

fly_spatial_parameters = FlySpatialParameters(orientation=0, position=np.array([0, 0]))

odor_history = OdorHistory()

odor_plume_all_ones = OdorPlumeAllOnes()

odor_plume_all_zeros = OdorPlumeAllZeros()

odor_plume_alternating = OdorPlumeAlternating()

wind_directions = WindDirections()


def plume_navigation_environment_plume_all_ones_simple_odor_hist_reward_factory():
    return PlumeNavigationEnvironment(wind_directions=wind_directions,
                                      fly_spatial_parameters=fly_spatial_parameters,
                                      odor_history=odor_history,
                                      odor_plume=odor_plume_all_ones,
                                      reward_flag=RewardSchemeEnum.SIMPLE_ODOR_HISTORY)


def plume_navigation_environment_plume_all_zeros_simple_odor_hist_reward_factory():
    return PlumeNavigationEnvironment(wind_directions=wind_directions,
                                      fly_spatial_parameters=fly_spatial_parameters,
                                      odor_history=odor_history,
                                      odor_plume=odor_plume_all_zeros,
                                      reward_flag=RewardSchemeEnum.SIMPLE_ODOR_HISTORY)


def plume_navigation_environment_plume_alternating_simple_odor_hist_reward_factory():
    return PlumeNavigationEnvironment(wind_directions=wind_directions,
                                      fly_spatial_parameters=fly_spatial_parameters,
                                      odor_history=odor_history,
                                      odor_plume=odor_plume_alternating,
                                      reward_flag=RewardSchemeEnum.SIMPLE_ODOR_HISTORY)


def plume_navigation_environment_plume_alternating_goal_zone_reward_factory():
    return PlumeNavigationEnvironment(wind_directions=wind_directions,
                                      fly_spatial_parameters=fly_spatial_parameters,
                                      odor_history=odor_history,
                                      odor_plume=odor_plume_alternating,
                                      reward_flag=RewardSchemeEnum.GOAL_ZONE)
