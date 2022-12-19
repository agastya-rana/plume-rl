from typing import Optional, Union, Tuple

import numpy as np
from gym import Env
from gym.core import ObsType, ActType
from gym.spaces import MultiDiscrete, Discrete
from matplotlib import pyplot as plt

from src.models.action_definitions import WalkActionEnum, WalkDisplacements
from src.models.fly_spatial_parameters import FlySpatialParameters
from src.models.goals import GoalZone
from src.models.odor_plumes import OdorMotionPlume, PLUME_VIDEO_X_BOUNDS, PLUME_VIDEO_Y_BOUNDS
from src.models.odor_senses import OdorFeatures, CONCENTRATION_THRESHOLD
from src.models.reward_schemes import RewardScheme, RewardSchemeEnum, \
    GoalZoneRewardScheme, SimpleOdorFeatureRewardScheme, NiragRewardScheme, YMaxRewardScheme
from src.models.wind_directions import WindDirections


class PlumeMotionNavigationEnvironment(Env):
    """
    This is structured like the OpenAI Gym Environments, with reset and step actions.
    When it is initialized, it takes a bunch of parameters that define how rewards are
    calculated, what the odor plume is, what the agent senses (odor features).
    Actual instances of this class are composed in motion_environment_factory.py
    An environment is a collection of a fly spatial instance, an odor plume instance,
    and an odor features instance. In addition, it specifies the reset and step actions
    """

    def __init__(self,
                 wind_directions: WindDirections,
                 fly_spatial_parameters: FlySpatialParameters,
                 odor_features: OdorFeatures,
                 odor_plume: OdorMotionPlume,
                 reward_flag: RewardSchemeEnum,
                 source_radius: float = 100):
        self.prior_frame = None
        self.wind_directions = wind_directions  # given as input
        self.fly_spatial_parameters = fly_spatial_parameters  # given as input
        self.odor_features = odor_features  # given as input MAKE DISCRETE
        self.odor_plume = odor_plume  # given as input
        self.reward_flag = reward_flag  # given as input, member of reward_enum tells how reward works
        self.observation_space = MultiDiscrete([2, 3, 3])  # This should go in the factory?
        self.action_space = Discrete(8)  # This should go in the factory?
        self.max_frames = 5000  # from Nirag real plume
        self.source_radius = source_radius
        self.x_bounds = PLUME_VIDEO_X_BOUNDS  # This doesn't belong here
        self.y_bounds = PLUME_VIDEO_Y_BOUNDS  # move to the plume itself
        self.current_trial_walk_displacement = np.array([0, 0])
        # self.render_fig = plt.figure()
        # self.render_ax = self.render_fig.add_subplot(111)

    def reset(
            self,
            *,
            seed: Optional[int] = 1234,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        """
        Reinialize the plume and the agent. Return the agent's initial observation.
        """
        rng: np.random.Generator = np.random.default_rng(seed)
        try:
            self.odor_plume.reset(flip=options['flip'])
        except TypeError:
            self.odor_plume.reset()

        self.prior_frame = self.odor_plume.get_previous_frame()
        odor_on = self.odor_plume.frame > CONCENTRATION_THRESHOLD
        odor_on_indices = np.transpose(odor_on.nonzero())

        try:
            x_random_bounds = options['randomization_x_bounds']
            y_random_bounds = options['randomization_y_bounds']
            self.fly_spatial_parameters.randomize_position(rng=rng,
                                                           x_bounds=x_random_bounds,
                                                           y_bounds=y_random_bounds,
                                                           valid_locations=odor_on_indices)
        except TypeError:
            print('Got no randomization bounds')
            self.fly_spatial_parameters.randomize_position(rng=rng)
        except KeyError:
            print("No bounds on randomizing fly position; using defaults")
            self.fly_spatial_parameters.randomize_position(rng=rng)

        self.fly_spatial_parameters.randomize_orientation(rng=rng)

        self.odor_features.clear()
        self.odor_features.update(sensor_location=self.fly_spatial_parameters.position,
                                  odor_plume_frame=self.odor_plume.frame,
                                  prior_odor_plume_frame=self.prior_frame)

        return self.odor_features.discretize_features()  # Change to a method that is implemented in all feature types

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """
        Advance the environment, based on an action selected by the agent.
        Return the next observation, the reward, a boolean indicating whether
        the trial is over, and a dictionary of any additional info that might be needed
        """
        walk_action = WalkActionEnum(action)
        walk_displacement = self.walk_functions[walk_action]
        self.current_trial_walk_displacement = walk_displacement  # This can be removed
        self.fly_spatial_parameters.update_position(walk_displacement)
        self.prior_frame = self.odor_plume.frame
        self.odor_plume.advance()
        self.odor_features.update(sensor_location=self.fly_spatial_parameters.position,
                                  odor_plume_frame=self.odor_plume.frame,
                                  prior_odor_plume_frame=self.prior_frame)
        reward = self.reward

        if self.odor_plume.frame_number >= self.max_frames:
            done = True
        else:
            done = False

        if self.fly_spatial_parameters.distance(distance_from=self.odor_plume.source_location) <= self.source_radius:
            done = True

        # observation, reward, done, info
        return self.odor_features.discretize_features(), \
               reward, \
               done, \
               {'concentration': self.odor_features.concentration,
                'motion_speed': self.odor_features.motion_speed,
                'gradient': self.odor_features.gradient}

    def render(self, mode="human"):
        """
        This is a method of openAI Gym Environments; but it's not yet implemented
        """
        assert mode in ["human", "rgb_array"], "Invalid mode: must be \'human\' or \'rgb_array\'"

    @property
    def walk_functions(self):
        walk_functions = WalkDisplacements(wind_params=self.wind_directions).walk_displacements
        return walk_functions

    @property
    def reward(self) -> float:  # This conditional thing should probably be selected at the model factory
        if self.reward_flag is RewardSchemeEnum.SIMPLE_ODOR_FEATURES:
            reward_scheme: RewardScheme = SimpleOdorFeatureRewardScheme(odor_features=self.odor_features)
        elif self.reward_flag is RewardSchemeEnum.GOAL_ZONE:
            reward_scheme: RewardScheme = GoalZoneRewardScheme(goal_zone=GoalZone(),
                                                               test_position=self.fly_spatial_parameters.position)
        elif self.reward_flag is RewardSchemeEnum.Y_MAX:
            reward_scheme: RewardScheme = YMaxRewardScheme(y_displacement=self.current_trial_walk_displacement[1])
        else:
            reward_scheme: RewardScheme = NiragRewardScheme(odor_features=self.odor_features)
        return reward_scheme.get_reward()
