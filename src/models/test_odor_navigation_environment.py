import numpy as np
import pytest

from src.models.environment_factory import plume_navigation_environment_plume_all_ones_simple_odor_hist_reward_factory, \
    plume_navigation_environment_plume_all_zeros_simple_odor_hist_reward_factory, \
    plume_navigation_environment_plume_alternating_simple_odor_hist_reward_factory, \
    plume_navigation_environment_plume_alternating_goal_zone_reward_factory
from src.models.odor_plumes import PLUME_VIDEO_Y_BOUNDS, PLUME_VIDEO_X_BOUNDS


@pytest.fixture()
def plume_navigation_environment_plume_all_ones_simple_odor_hist_reward():
    return plume_navigation_environment_plume_all_ones_simple_odor_hist_reward_factory()

@pytest.fixture()
def plume_navigation_environment_plume_all_zeros_simple_odor_hist_reward():
    return plume_navigation_environment_plume_all_zeros_simple_odor_hist_reward_factory()

@pytest.fixture()
def plume_navigation_environment_plume_alternating_simple_odor_hist_reward():
    return plume_navigation_environment_plume_alternating_simple_odor_hist_reward_factory()

@pytest.fixture()
def plume_navigation_environment_plume_alternating_goal_zone_reward_factor():
    return plume_navigation_environment_plume_alternating_goal_zone_reward_factory()


def test_can_declare_box(
        plume_navigation_environment_plume_all_ones_simple_odor_hist_reward):
    my_environment = plume_navigation_environment_plume_all_ones_simple_odor_hist_reward
    assert my_environment is not None



def test_reset_plume_nav_env_should_randomize_fly_near_plume(
        plume_navigation_environment_plume_all_ones_simple_odor_hist_reward):
    plume_x_bounds: np.ndarray = PLUME_VIDEO_X_BOUNDS # Bounds taken from Nirag
    plume_y_bounds = np.ndarray = PLUME_VIDEO_Y_BOUNDS  # Bounds taken from Nirag

    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.reset(seed=42)
    assert (plume_x_bounds[0] <
            plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.fly_spatial_parameters.position[0]) & \
           (plume_x_bounds[1] >
            plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.fly_spatial_parameters.position[0]) & \
           (plume_y_bounds[0] <
            plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.fly_spatial_parameters.position[1]) & \
           (plume_y_bounds[1] >
            plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.fly_spatial_parameters.position[1])

"""
def test_reset_plume_nav_env_should_return_an_odor_history_vector(
        plume_navigation_environment_plume_all_ones_simple_odor_hist_reward):
    observation = plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.reset()
    assert len(observation) == MAX_HISTORY_LENGTH


def test_in_plume_starting_with_ones_observation_should_have_a_one_at_the_end_after_environment_resets(
        plume_navigation_environment_plume_all_ones_simple_odor_hist_reward):
    observation = plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.reset()
    assert observation[-1] == 1


def test_step_upwind_turn_from_facing_east_in_northerly_wind_should_yield_orientation_at_one_sixth_of_a_full_turn_ccw(
        plume_navigation_environment_plume_all_ones_simple_odor_hist_reward):
    action = TurnActionEnum.UPWIND_TURN.value
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.wind_directions = WindDirections(3 * np.pi / 2)
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.fly_spatial_parameters.orientation = 0
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.step(action)
    new_orientation = \
        plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.fly_spatial_parameters.orientation
    expected_orientation = np.pi / 6
    assert new_orientation == expected_orientation


def test_in_all_ones_plume_an_observation_following_reset_and_step_should_end_in_a_pair_of_ones(
        plume_navigation_environment_plume_all_ones_simple_odor_hist_reward):
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.reset()
    action = TurnActionEnum.UPWIND_TURN.value
    observation, reward, done, info = plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.step(action)
    assert np.array_equal(observation[-3:], np.array([0, 1, 1]))


def test_in_alternating_plume_starting_with_ones_an_observation_following_reset_and_step_should_end_in_010(
        plume_navigation_environment_plume_all_ones_simple_odor_hist_reward):
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.odor_plume = OdorPlumeAlternating()
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.reset()
    action = TurnActionEnum.UPWIND_TURN.value
    observation, reward, done, info = plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.step(action)
    assert np.array_equal(observation[-3:], np.array([0, 1, 0]))


def test_all_ones_plume_and_simple_odor_reward_scheme_should_yield_reward_of_one_after_first_step(
        plume_navigation_environment_plume_all_ones_simple_odor_hist_reward):
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.reset()
    action = TurnActionEnum.UPWIND_TURN.value
    _, reward, _, _ = plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.step(action)
    expected_reward = 1
    assert reward == expected_reward


def test_all_zeros_plume_and_simple_odor_reward_scheme_should_yield_reward_of_zero_after_first_step(
        plume_navigation_environment_plume_all_ones_simple_odor_hist_reward):
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.odor_plume = OdorPlumeAllZeros()
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.reset()
    action = TurnActionEnum.UPWIND_TURN.value
    _, reward, _, _ = plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.step(action)
    expected_reward = 0
    assert reward == expected_reward


def test_fly_outside_goal_zone_and_goal_zone_reward_scheme_should_yield_reward_of_zero(
        plume_navigation_environment_plume_all_ones_simple_odor_hist_reward):
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.reward_flag = RewardSchemeEnum.GOAL_ZONE
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.reset()
    action = TurnActionEnum.UPWIND_TURN.value
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.fly_spatial_parameters.position = \
        np.array([GOAL_X + GOAL_RADIUS + 100, GOAL_Y + GOAL_RADIUS + 100])
    _, reward, _, _ = plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.step(action)
    expected_reward = 0
    assert reward == expected_reward


def test_fly_inside_goal_zone_and_goal_zone_reward_scheme_should_yield_reward_above_zero(
        plume_navigation_environment_plume_all_ones_simple_odor_hist_reward):
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.reward_flag = RewardSchemeEnum.GOAL_ZONE
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.reset()
    action = TurnActionEnum.UPWIND_TURN.value
    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.fly_spatial_parameters.position = \
        np.array([GOAL_X, GOAL_Y])
    _, reward, _, _ = plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.step(action)
    assert reward > 0
"""
