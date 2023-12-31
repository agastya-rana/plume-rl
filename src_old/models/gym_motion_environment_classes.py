from enum import Enum
from typing import Optional, Union, Tuple, Type

import numpy as np
from gym import Env
from gym.core import ObsType, ActType
from gym.spaces import MultiDiscrete, Discrete

from src.models.action_definitions import *
from src.models.fly_spatial_parameters import FlySpatialParameters
from src.models.goals import GoalZone
from src.models.integrator_senses import IntegratorSensor
from src.models.odor_plumes import * #PLUME_VIDEO_X_BOUNDS, PLUME_VIDEO_Y_BOUNDS
from src.models.odor_senses import *
from src.models.reward_schemes import *
from src.models.wind_directions import WindDirections


class PlumeMotionNavigationEnvironment(Env):
    """
    This is structured like the OpenAI Gym Environments, with reset and step actions.
    When it is initialized, it takes a bunch of parameters that define how rewards are
    calculated, what the odor plume is, what the agent senses (odor features).
    Actual instances of this class are composed in motion_environment_factory.py
    This class specifies the reset and step actions that are needed for openAI gym.
    The 'render' method is part of Gym environments but isn't implemented yet.
    """

    def __init__(self, 
                 rng,
                 config,
                 wind_directions: WindDirections,
                 fly_spatial_parameters: FlySpatialParameters,
                 odor_features: OdorFeatures,
                 odor_plume: OdorPlumeFromMovie,
                 reward_flag: RewardSchemeEnum,
                 action_enum: Union[Type[WalkActionEnum], Type[WalkStopActionEnum]] = WalkActionEnum,  # default is
                 # consistent with original design
                 action_class=WalkDisplacements,
                 ):
        self.prior_frame = None
        self.wind_directions = wind_directions  # given as input
        self.fly_spatial_parameters = fly_spatial_parameters  # given as input
        self.odor_features = odor_features
        self.odor_plume = odor_plume  # given as input
        self.reward_flag = reward_flag  # given as input, member of reward_enum tells how reward works
        self.observation_space = MultiDiscrete([2, 3, 3])  # This should go in the factory?
        self.action_space = Discrete(len(action_enum))
        self.action_enum = action_enum
        self.action_class: DisplacementActionClass = action_class
        self.source_radius = config['GOAL_RADIUS_MM']
        self.current_trial_walk_displacement = np.array([0, 0])
        self.rng = rng
        self.config = config
        self.max_frames = config['STOP_FRAME']
        # self.render_fig = plt.figure()
        # self.render_ax = self.render_fig.add_subplot(111)

    def reset(
            self,
            *,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        """
        Reinitialize the plume and the agent. Return the agent's initial observation.
        """
        #rng: np.random.Generator = np.random.default_rng(seed)
        # Step 1: get odor frame
        try:
            self.odor_plume.reset(flip=options['flip'], rng = self.rng)
        except TypeError:
            self.odor_plume.reset()

        self.prior_frame = self.odor_plume.get_previous_frame()
        odor_on = self.odor_plume.frame > self.config['CONCENTRATION_THRESHOLD']
        odor_on_indices = np.transpose(odor_on.nonzero())

        # Step 2: Initialize fly position
        try:
            x_random_bounds = options['randomization_x_bounds']
            y_random_bounds = options['randomization_y_bounds']

            valid_locations = odor_on_indices*self.config['MM_PER_PX']

            self.fly_spatial_parameters.randomize_position(rng=self.rng,
                                                           x_bounds=x_random_bounds,
                                                           y_bounds=y_random_bounds,
                                                           valid_locations=valid_locations)
        except TypeError:
            print('Got no randomization bounds')

            min_x = self.config['MIN_RESET_X_MM']
            max_x = self.config['MAX_RESET_X_MM']

            x_bounds = np.array([min_x, max_x])


            min_y = self.config['MIN_RESET_Y_MM']
            max_y = self.config['MAX_RESET_Y_MM']

            y_bounds = np.array([min_y, max_y])

            self.fly_spatial_parameters.randomize_position(rng=self.rng, x_bounds = x_bounds, y_bounds = y_bounds)
        
        #except KeyError:
        #    print("No bounds on randomizing fly position; using defaults")
        #   self.fly_spatial_parameters.randomize_position(rng=self.rng)

        #self.fly_spatial_parameters.randomize_orientation(rng=self.rng)

        # Smell
        self.odor_features.clear()
        self.odor_features.update(config = self.config, sensor_location=self.fly_spatial_parameters.position,
                                  odor_plume_frame=self.odor_plume.frame,
                                  prior_odor_plume_frame=self.prior_frame)

        return self.odor_features.discretize_features(config = self.config)  # Change to a method that is implemented in all feature types

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """
        Advance the environment, based on an action selected by the agent.
        Return the next observation, the reward, a boolean indicating whether
        the trial is over, and a dictionary of any additional info that might be needed
        """
        # Translate action level (i.e., integer denoting action) into a spatial displacement
        walk_action = self.action_enum(action)
        walk_displacement = self.walk_functions[walk_action]

        self.current_trial_walk_displacement = walk_displacement  # This can be removed

        # Update fly position using calculated action displacement
        self.fly_spatial_parameters.update_position(walk_displacement)

        # Advance odor plume
        self.prior_frame = self.odor_plume.frame

        self.odor_plume.advance(rng = self.rng)

        # Sniff
        self.odor_features.update(sensor_location=self.fly_spatial_parameters.position,
                                  odor_plume_frame=self.odor_plume.frame,
                                  prior_odor_plume_frame=self.prior_frame, config = self.config)

        reward = self.reward # this should be calculate_reward() method

        if self.odor_plume.frame_number >= self.max_frames:
            done = True
        else:
            done = False

        # add a punishment to discourage always walking upwind
        if self.fly_spatial_parameters.position[0] < self.config['WALL_MIN_X_MM']:
            reward = -10
            done = True
        elif self.fly_spatial_parameters.position[0] > self.config['WALL_MAX_X_MM']:
            reward = -10
            done = True
        elif self.fly_spatial_parameters.position[1]<self.config['WALL_MIN_Y_MM']:
            reward = -10
            done = True
        elif self.fly_spatial_parameters.position[1] > self.config['WALL_MAX_Y_MM']:
            reward = -10
            done = True


        if self.fly_spatial_parameters.distance(distance_from=self.odor_plume.source_location) <= self.source_radius:
            done = True



        # observation, reward, done, info
        return self.odor_features.discretize_features(config = self.config), \
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
    def walk_functions(self) -> dict[Enum, np.ndarray]:
        actions = self.action_class(wind_params=self.wind_directions, config = self.config)
        walk_functions = actions.walk_displacements
        return walk_functions

    @property
    def reward(self) -> float:  # This conditional thing should probably be selected at the model factory
        if self.reward_flag is RewardSchemeEnum.SIMPLE_ODOR_FEATURES:
            reward_scheme: RewardScheme = SimpleOdorFeatureRewardScheme(odor_features=self.odor_features)
        elif self.reward_flag is RewardSchemeEnum.GOAL_ZONE:
            reward_scheme: RewardScheme = GoalZoneRewardScheme(goal_zone=GoalZone(self.config),
                                                               test_position=self.fly_spatial_parameters.position)
        elif self.reward_flag is RewardSchemeEnum.Y_MAX:
            reward_scheme: RewardScheme = YMaxRewardScheme(y_displacement=self.current_trial_walk_displacement[1])
        else:
            reward_scheme: RewardScheme = NiragRewardScheme(odor_features=self.odor_features)
        return reward_scheme.get_reward()



