import numpy as np
import pytest

from src.models.action_definitions import WalkStopActionEnum, WalkDisplacements
from src.models.wind_directions import WindDirections


@pytest.fixture
def easterly_walk_displacements():
    return WalkDisplacements(WindDirections(wind_angle=0))


@pytest.fixture
def westerly_walk_displacements():
    return WalkDisplacements(WindDirections(wind_angle=np.pi))


def test_downwind_in_easterly_wind_should_be_5_0_displacement(easterly_walk_displacements):
    expected_displacement = [5, 0]
    calculated_displacement = easterly_walk_displacements.walk_displacements[WalkStopActionEnum.DOWNWIND]
    assert np.allclose(expected_displacement, calculated_displacement)


def test_upwind_in_easterly_wind_should_be_n5_0_displacement(easterly_walk_displacements):
    expected_displacement = [-5, 0]
    calculated_displacement = easterly_walk_displacements.walk_displacements[WalkStopActionEnum.UPWIND]
    assert np.allclose(expected_displacement, calculated_displacement)


def test_upwind_in_westerly_wind_should_be_5_0_displacement(westerly_walk_displacements):
    expected_displacement = [5, 0]
    calculated_displacement = westerly_walk_displacements.walk_displacements[WalkStopActionEnum.UPWIND]
    assert np.allclose(expected_displacement, calculated_displacement)
