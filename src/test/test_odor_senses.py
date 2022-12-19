import numpy as np
import pytest

from src.models.fly_spatial_parameters import FlySpatialParameters
from src.models.odor_plumes import OdorPlumeAllOnes
from src.models.odor_senses import detect_local_odor_concentration, MAX_HISTORY_LENGTH, OdorHistory, measure_odor_speed


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


def test_motion_sensor_should_detect_1step_rightward_roll_as_speed1():
    odor_array_0 = np.arange(1, 15)  # 14 pixels from Nirag's paper
    odor_array_1 = np.roll(odor_array_0, 1)
    delta_x_hat = measure_odor_speed(odor_array_0, odor_array_1)
    expected_speed = 1
    assert delta_x_hat == expected_speed


def test_motion_sensor_should_detect_1step_leftward_roll_as_speedn1():
    odor_array_0 = np.arange(1, 15)  # 14 pixels from Nirag's paper
    odor_array_1 = np.roll(odor_array_0, -1)
    delta_x_hat = measure_odor_speed(odor_array_0, odor_array_1)
    expected_speed = -1
    assert delta_x_hat == expected_speed


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
