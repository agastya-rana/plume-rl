import numpy as np
import pytest

from src.models.action_definitions import TurnActionEnum

from src.models.environment_factory import \
    PlumeNavigationEnvironmentPlumeAllOnesSimpleOdorHistRewardWesterlyWindFactory, \
    PlumeNavigationEnvironmentPlumeAllOnesSimpleOdorHistRewardNortherlyWindFactory, \
    PlumeNavigationEnvironmentAlternatingPlumeSimpleOdorHistRewardWesterlyWindFactory, \
    PlumeNavigationEnvironmentPlumeZerosSimpleOdorHistRewardWesterlyWindFactory, \
    PlumeNavigationEnvironmentPlumeAlternatingGoalZoneRewardWesterlyWindFactory
from src.models.goals import GOAL_X, GOAL_RADIUS, GOAL_Y
from src.models.odor_histories import MAX_HISTORY_LENGTH
from src.models.odor_plumes import PLUME_VIDEO_Y_BOUNDS, PLUME_VIDEO_X_BOUNDS


class TestPlumeNavigationEnvironmentPlumeAllOnesSimpleOdorHistRewardWesterlyWind:
    plume_env = PlumeNavigationEnvironmentPlumeAllOnesSimpleOdorHistRewardWesterlyWindFactory().plume_environment

    @staticmethod
    def _in_bounds(position):
        plume_x_bounds: np.ndarray = PLUME_VIDEO_X_BOUNDS  # Bounds taken from Nirag
        plume_y_bounds = np.ndarray = PLUME_VIDEO_Y_BOUNDS  # Bounds taken from Nirag
        in_bounds = (plume_x_bounds[0] <
                     position[0]) & \
                    (plume_x_bounds[1] >
                     position[0]) & \
                    (plume_y_bounds[0] <
                     position[1]) & \
                    (plume_y_bounds[1] >
                     position[1])
        return in_bounds

    def test_reset_plume_nav_env_should_randomize_fly_near_plume(self):
        print(self.plume_env)
        self.plume_env.reset(seed=42)
        assert self._in_bounds(self.plume_env.fly_spatial_parameters.position)

    def test_reset_plume_nav_env_should_return_an_odor_history_vector(self):
        observation = self.plume_env.reset(seed=42)
        assert len(observation) == MAX_HISTORY_LENGTH

    def test_observation_should_have_a_one_at_the_end_after_environment_resets(self):
        observation = self.plume_env.reset(seed=42)
        assert observation[-1] == 1

    def test_an_observation_following_reset_and_step_should_end_in_a_pair_of_ones(self):
        self.plume_env.reset(seed=42)
        action = TurnActionEnum.UPWIND_TURN.value
        observation, reward, done, info = self.plume_env.step(action)
        assert np.array_equal(observation[-3:], np.array([0, 1, 1]))

    def test_reward_should_be_one_after_first_step(self):
        self.plume_env.reset(seed=42)
        action = TurnActionEnum.UPWIND_TURN.value
        _, reward, _, _ = self.plume_env.step(action)
        expected_reward = 1
        assert reward == expected_reward


class TestPlumeNavigationEnvironmentPlumeAllOnesSimpleOdorHistRewardNortherlyWind:
    plume_env = PlumeNavigationEnvironmentPlumeAllOnesSimpleOdorHistRewardNortherlyWindFactory().plume_environment

    def test_step_upwind_turn_from_facing_east_should_yield_orientation_at_one_sixth_of_a_full_turn_ccw(self):
        action = TurnActionEnum.UPWIND_TURN.value
        self.plume_env.fly_spatial_parameters.orientation = 0
        self.plume_env.step(action)
        new_orientation = \
            self.plume_env.fly_spatial_parameters.orientation
        expected_orientation = np.pi / 6
        assert new_orientation == expected_orientation


class TestPlumeNavigationEnvironmentAlternatingPlumeSimpleOdorHistRewardNortherlyWind:
    plume_env = PlumeNavigationEnvironmentAlternatingPlumeSimpleOdorHistRewardWesterlyWindFactory().plume_environment

    def test_an_observation_following_reset_and_step_should_end_in_010(self):
        self.plume_env.reset(seed=42)
        action = TurnActionEnum.UPWIND_TURN.value
        observation, reward, done, info = self.plume_env.step(action)
        assert np.array_equal(observation[-3:], np.array([0, 1, 0]))


class TestPlumeNavigationEnvironmentPlumeAllZerosSimpleOdorHistRewardWesterlyWind:
    plume_env = PlumeNavigationEnvironmentPlumeZerosSimpleOdorHistRewardWesterlyWindFactory().plume_environment

    def test_reward_should_be_zero_after_first_step(self):
        self.plume_env.reset(seed=42)
        action = TurnActionEnum.UPWIND_TURN.value
        _, reward, _, _ = self.plume_env.step(action)
        expected_reward = 0
        assert reward == expected_reward


class TestPlumeNavigationEnvironmentPlumeAlternatingSimpleOdorHistRewardWesterlyWind:
    plume_env = PlumeNavigationEnvironmentPlumeAlternatingGoalZoneRewardWesterlyWindFactory().plume_environment

    def test_fly_outside_goal_zone_and_goal_zone_reward_scheme_should_yield_reward_of_zero(self):
        self.plume_env.reset(seed=42)
        action = TurnActionEnum.UPWIND_TURN.value
        self.plume_env.fly_spatial_parameters.position = \
            np.array([GOAL_X + GOAL_RADIUS + 100, GOAL_Y + GOAL_RADIUS + 100])
        _, reward, _, _ = self.plume_env.step(action)
        expected_reward = 0
        assert reward == expected_reward

    def test_fly_inside_goal_zone_and_goal_zone_reward_scheme_should_yield_reward_above_zero(self):
        self.plume_env.reset()
        action = TurnActionEnum.UPWIND_TURN.value
        self.plume_env.fly_spatial_parameters.position = \
            np.array([GOAL_X, GOAL_Y])
        _, reward, _, _ = self.plume_env.step(action)
        assert reward > 0
