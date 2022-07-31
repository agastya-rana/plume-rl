import numpy as np
import pytest

from src.models.odor_navigation_environment import FlySpatialParameters, MAX_HISTORY_LENGTH, \
    detect_local_odor_concentration, OdorHistory, WindDirections, GOAL_X, GOAL_Y, GOAL_RADIUS
from src.models.reward_schemes import SimpleOdorHistoryRewardScheme, RewardSchemeEnum
from src.models.action_definitions import TurnActionEnum, TurnFunctions
from src.models.gym_environment_classes import PlumeNavigationEnvironment
from src.models.geometry import standardize_angle, angle_to_unit_vector
from src.models.odor_plumes import PLUME_VIDEO_Y_BOUNDS, PLUME_VIDEO_X_BOUNDS, OdorPlumeAllOnes, OdorPlumeAlternating, \
    OdorPlumeAllZeros


@pytest.fixture
def fly_spatial_parameters():
    return FlySpatialParameters(orientation=0, position=np.array([0, 0]))


@pytest.fixture
def max_history_length():
    return MAX_HISTORY_LENGTH


@pytest.fixture
def odor_history():
    return OdorHistory()


@pytest.fixture
def odor_plume_all_ones():
    return OdorPlumeAllOnes()


@pytest.fixture()
def plume_navigation_environment_plume_all_ones_simple_odor_hist_reward(fly_spatial_parameters,
                                                                        odor_history,
                                                                        odor_plume_all_ones):
    wind_directions = WindDirections()
    return PlumeNavigationEnvironment(wind_directions=wind_directions,
                                      fly_spatial_parameters=fly_spatial_parameters,
                                      odor_history=odor_history,
                                      odor_plume=odor_plume_all_ones,
                                      reward_flag=RewardSchemeEnum.SIMPLE_ODOR_HISTORY)


def test_fly_should_be_north_of_starting_position_after_walking_north_from_starting_position(fly_spatial_parameters):
    walking_direction = np.array([0, 1])
    fly_spatial_parameters.update_position(walking_direction)
    expected_position = np.array([0, 1])
    assert np.array_equal(fly_spatial_parameters.position, expected_position)


def test_fly_should_be_south_of_starting_position_after_walking_south_from_starting_position(fly_spatial_parameters):
    walking_direction = np.array([0, -1])
    fly_spatial_parameters.update_position(walking_direction)
    expected_position = np.array([0, -1])
    assert np.array_equal(fly_spatial_parameters.position, expected_position)


def test_fly_should_be_east_of_starting_position_after_walking_east_from_starting_position(fly_spatial_parameters):
    walking_direction = np.array([1, 0])
    fly_spatial_parameters.update_position(walking_direction)
    expected_position = np.array([1, 0])
    assert np.array_equal(fly_spatial_parameters.position, expected_position)


def test_fly_should_face_north_after_quarter_turn_ccw(fly_spatial_parameters):
    turn_angle = np.pi / 2
    fly_spatial_parameters.turn(turn_angle)
    expected_angle = np.pi / 2
    assert np.array_equal(fly_spatial_parameters.orientation, expected_angle)


def test_fly_should_face_south_after_quarter_turn_cw(fly_spatial_parameters):
    turn_angle = - np.pi / 2
    fly_spatial_parameters.turn(turn_angle)
    expected_angle = - np.pi / 2
    expected_angle = standardize_angle(expected_angle)
    assert np.array_equal(fly_spatial_parameters.orientation, expected_angle)


def test_fly_should_face_south_after_three_quarters_turn_ccw(fly_spatial_parameters):
    turn_angle = np.pi * 3 / 2
    fly_spatial_parameters.turn(turn_angle)
    expected_angle = - np.pi / 2
    expected_angle = standardize_angle(expected_angle)
    assert np.array_equal(fly_spatial_parameters.orientation, expected_angle)


def test_fly_should_be_north_after_turning_north_and_walking_forward(fly_spatial_parameters):
    turn_angle = np.pi / 2
    fly_spatial_parameters.turn_and_walk(turn_angle)
    expected_position = np.array([0, 1])
    assert np.allclose(fly_spatial_parameters.position, expected_position)


def test_fly_should_be_south_after_turning_south_and_walking_forward(fly_spatial_parameters):
    turn_angle = - np.pi / 2
    fly_spatial_parameters.turn_and_walk(turn_angle)
    expected_position = np.array([0, -1])
    assert np.allclose(fly_spatial_parameters.position, expected_position)


def test_fly_should_face_south_after_upwind_quarter_turn_from_zero_in_southerly_wind(fly_spatial_parameters):
    wind_direction = np.pi / 2
    turn_functions = TurnFunctions(wind_params=WindDirections(wind_angle=wind_direction))
    turn_sign = turn_functions.turn_against_orientation_sign(fly_spatial_parameters.orientation, wind_direction)
    quarter_turn_magnitude = np.pi / 2
    turn_angle = turn_sign * quarter_turn_magnitude
    fly_spatial_parameters.turn(turn_angle)
    expected_angle = 3 * np.pi / 2
    assert np.array_equal(fly_spatial_parameters.orientation, expected_angle)


def test_fly_should_face_north_after_upwind_quarter_turn_from_zero_in_northerly_wind(fly_spatial_parameters):
    wind_direction = 3 * np.pi / 2
    turn_functions = TurnFunctions(wind_params=WindDirections(wind_angle=wind_direction))
    turn_sign = turn_functions.turn_against_orientation_sign(fly_spatial_parameters.orientation, wind_direction)
    quarter_turn_magnitude = np.pi / 2
    fly_spatial_parameters.turn(turn_sign * quarter_turn_magnitude)
    expected_angle = np.pi / 2
    assert np.array_equal(fly_spatial_parameters.orientation, expected_angle)


def test_odor1_everywhere_should_drive_local_odor_concentration_to_be_1(fly_spatial_parameters):
    random_position = np.array([12.3, 98.7])
    fly_spatial_parameters.position = random_position
    odor_plume_frame = np.ones([100, 100])
    local_odor_concentration = detect_local_odor_concentration(fly_spatial_parameters.position, odor_plume_frame)
    assert local_odor_concentration == 1


def test_fly_should_update_odor_history_using_local_odor_concentration(fly_spatial_parameters, odor_history):
    random_position = np.array([12.3, 98.7])
    fly_spatial_parameters.position = random_position
    odor_plume_frame = np.ones([100, 100])
    local_odor_concentration = detect_local_odor_concentration(fly_spatial_parameters.position, odor_plume_frame)
    odor_history.update(fly_spatial_parameters.position, odor_plume_frame)
    assert odor_history.value[-1] == local_odor_concentration


def test_updating_odor_history_should_preserve_odor_history_length(fly_spatial_parameters, odor_history):
    random_position = np.array([12.3, 98.7])
    fly_spatial_parameters.position = random_position
    odor_plume_frame = np.ones([100, 100])
    start_size = odor_history.value.size
    odor_history.update(fly_spatial_parameters.position, odor_plume_frame)
    end_size = odor_history.value.size
    assert start_size == end_size


def test_crosswind_direction_a_should_be_orthogonal_to_wind():
    wind_directions = WindDirections(3 * np.pi / 2)
    wind_direction = angle_to_unit_vector(wind_directions.wind_angle)
    crosswind_direction_a = angle_to_unit_vector(wind_directions.crosswind_a)
    assert np.dot(wind_direction, crosswind_direction_a) == pytest.approx(0)


def test_crosswind_direction_b_should_be_orthogonal_to_wind():
    wind_directions = WindDirections(3 * np.pi / 2)
    wind_direction = angle_to_unit_vector(wind_directions.wind_angle)
    crosswind_direction_b = angle_to_unit_vector(wind_directions.crosswind_b)
    assert np.dot(wind_direction, crosswind_direction_b) == pytest.approx(0)


def test_action_no_turn_should_leave_orientation_alone(fly_spatial_parameters):
    wind_directions = WindDirections(3 * np.pi / 2)
    turn_functions = TurnFunctions(wind_params=wind_directions)
    fly_spatial_parameters.orientation = 0
    fly_spatial_parameters.orientation = \
        turn_functions.turn_functions[TurnActionEnum.NO_TURN](fly_spatial_parameters.orientation)
    assert fly_spatial_parameters.orientation == 0


def test_action_upwind_turn_from_facing_east_in_northerly_wind_should_yield_orientation_at_one_sixth_of_a_full_turn_ccw(
        fly_spatial_parameters):
    wind_directions = WindDirections(3 * np.pi / 2)
    turn_funcs = TurnFunctions(wind_params=wind_directions).turn_functions
    fly_spatial_parameters.orientation = 0
    fly_spatial_parameters.orientation = \
        turn_funcs[TurnActionEnum.UPWIND_TURN](fly_spatial_parameters.orientation)
    expected_angle = np.pi / 6
    assert fly_spatial_parameters.orientation == expected_angle


def test_reset_plume_nav_env_should_randomize_fly_near_plume(
        plume_navigation_environment_plume_all_ones_simple_odor_hist_reward,
        fly_spatial_parameters):
    plume_x_bounds: np.ndarray = PLUME_VIDEO_X_BOUNDS  # Bounds taken from Nirag
    plume_y_bounds = np.ndarray = PLUME_VIDEO_Y_BOUNDS  # Bounds taken from Nirag

    plume_navigation_environment_plume_all_ones_simple_odor_hist_reward.reset(seed=42)
    assert (plume_x_bounds[0] < fly_spatial_parameters.position[0]) & \
           (plume_x_bounds[1] > fly_spatial_parameters.position[0]) & \
           (plume_y_bounds[0] < fly_spatial_parameters.position[1]) & \
           (plume_y_bounds[1] > fly_spatial_parameters.position[1])


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
