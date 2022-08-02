from typing import Optional, Union, Tuple

import numpy as np
from gym import Env, spaces
from gym.core import ObsType, ActType

from src.models.action_definitions import TurnActionEnum, TurnFunctions
from src.models.goals import GoalZone
from src.models.fly_spatial_parameters import FlySpatialParameters
from src.models.odor_histories import OdorHistory
from src.models.wind_directions import WindDirections
from src.models.odor_plumes import OdorPlume
from src.models.reward_schemes import RewardScheme, RewardSchemeEnum, SimpleOdorHistoryRewardScheme, \
    GoalZoneRewardScheme


class PlumeNavigationEnvironment(Env):
    """
    An environment is a collection of a fly spatial instance, an odor plume instance,
    and an odor history instance. In addition, it specifies the sequence of reset and
    step actions defined by these objects
    """

    def __init__(self,
                 wind_directions: WindDirections,
                 fly_spatial_parameters: FlySpatialParameters,
                 odor_history: OdorHistory,
                 odor_plume: OdorPlume,
                 reward_flag: RewardSchemeEnum):
        self.wind_directions = wind_directions
        self.fly_spatial_parameters = fly_spatial_parameters
        self.odor_history = odor_history
        self.odor_plume = odor_plume
        self.reward_flag = reward_flag
        self.observation_space = spaces.Box(low=0, high=1, shape=(1000,))

    def reset(
            self,
            *,
            seed: Optional[int] = 1234,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        rng: np.random.Generator = np.random.default_rng(seed)

        self.fly_spatial_parameters.randomize_position(rng=rng)
        self.fly_spatial_parameters.randomize_orientation(rng=rng)
        self.odor_plume.reset()
        self.odor_history.clear()
        self.odor_history.update(self.fly_spatial_parameters.position,
                                 self.odor_plume.frame)

        return self.odor_history.value

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        turn_action = TurnActionEnum(action)
        turn_angle = self.turn_functions[turn_action](self.fly_spatial_parameters.orientation)
        self.fly_spatial_parameters.turn_and_walk(turn_angle=turn_angle)
        self.odor_plume.advance()
        self.odor_history.update(self.fly_spatial_parameters.position, self.odor_plume.frame)
        reward = self.reward
        return self.odor_history.value, reward, False, {'data': None}  # observation, reward, done, info

    def render(self, mode="human"):
        pass

    @property
    def turn_functions(self):
        turn_functions = TurnFunctions(wind_params=self.wind_directions).turn_functions
        return turn_functions

    @property
    def reward(self) -> float:
        if self.reward_flag is RewardSchemeEnum.SIMPLE_ODOR_HISTORY:
            reward_scheme: RewardScheme = SimpleOdorHistoryRewardScheme(odor_history=self.odor_history)
        else:
            reward_scheme: RewardScheme = GoalZoneRewardScheme(goal_zone=GoalZone(),
                                                               test_position=self.fly_spatial_parameters.position)
        return reward_scheme.get_reward()
