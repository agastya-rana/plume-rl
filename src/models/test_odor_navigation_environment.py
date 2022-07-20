from src.models import odor_navigation_environment
import numpy as np
import pytest


class TestNoOdorCenterUpwindStationaryFlyOdorNavigationEnvironment:
    no_odor_center_upwind_stat_fly_env = \
        odor_navigation_environment.NoOdorCenterUpwindStationaryFlyOdorNavigationEnvironment()

    def test_should_initialize_fly_without_position(self):
        assert np.all(np.isnan(self.no_odor_center_upwind_stat_fly_env.fly.position))

    def test_should_initialize_fly_without_speed(self):
        assert np.isnan(self.no_odor_center_upwind_stat_fly_env.fly.speed)

    def test_should_initialize_fly_without_orientation(self):
        assert np.isnan(self.no_odor_center_upwind_stat_fly_env.fly.angle)

    def test_should_initialize_null_plume(self):
        assert np.all(np.isnan(self.no_odor_center_upwind_stat_fly_env.plume.snapshot))

    def test_should_reset_to_centered_fly(self):
        self.no_odor_center_upwind_stat_fly_env.reset()
        center = np.array([0, 0])
        assert np.array_equal(center, self.no_odor_center_upwind_stat_fly_env.fly.position)

    def test_should_reset_to_stationary_fly(self):
        self.no_odor_center_upwind_stat_fly_env.reset()
        stationary_speed = 0
        assert self.no_odor_center_upwind_stat_fly_env.fly.speed == stationary_speed

    def test_should_reset_to_upwind_fly(self):
        self.no_odor_center_upwind_stat_fly_env.reset()
        upwind_angle = 0
        assert self.no_odor_center_upwind_stat_fly_env.fly.angle == upwind_angle

    def test_should_reset_to_no_odor(self):
        self.no_odor_center_upwind_stat_fly_env.reset()
        assert np.all(self.no_odor_center_upwind_stat_fly_env.plume.snapshot == 0)


class TestAlternatingOdorPlume:
    alternating_odor_plume = odor_navigation_environment.AlternatingOdorPlume()

    def test_should_initialize_null_plume(self):
        assert np.all(np.isnan(self.alternating_odor_plume.snapshot))

    def test_should_reset_to_no_odor(self):
        self.alternating_odor_plume.reset()
        assert np.all(self.alternating_odor_plume.snapshot == 0)

    def test_first_step_should_flash_odor(self):
        self.alternating_odor_plume.reset()
        self.alternating_odor_plume.step()
        assert np.all(self.alternating_odor_plume.snapshot == 1)

    def test_second_step_should_flash_odor(self):
        self.alternating_odor_plume.reset()
        self.alternating_odor_plume.step()
        self.alternating_odor_plume.step()
        assert np.all(self.alternating_odor_plume.snapshot == 0)


class TestAcceleratingRotatingFly:
    #NB to make calculating values for tests easy, fly rotation is a quarter turn per time step
    accelerating_rotating_fly = odor_navigation_environment.AcceleratingRotatingFly()
    def test_should_initialize_fly_without_position(self):
        assert np.all(np.isnan(self.accelerating_rotating_fly.position))

    def test_should_initialize_fly_without_speed(self):
        assert np.isnan(self.accelerating_rotating_fly.speed)

    def test_should_initialize_fly_without_orientation(self):
        assert np.isnan(self.accelerating_rotating_fly.angle)

    def test_should_reset_to_centered_fly(self):
        self.accelerating_rotating_fly.reset()
        center = np.array([0, 0])
        assert np.array_equal(center, self.accelerating_rotating_fly.position)

    def test_should_reset_to_slow_fly(self):
        self.accelerating_rotating_fly.reset()
        slow_speed = 0.01
        assert self.accelerating_rotating_fly.speed == slow_speed

    def test_step_should_increment_speed(self):
        self.accelerating_rotating_fly.reset()
        self.accelerating_rotating_fly.step()
        speed_step2 = 0.02
        assert self.accelerating_rotating_fly.speed == speed_step2

    def test_should_reset_to_upwind_fly(self):
        self.accelerating_rotating_fly.reset()
        upwind_angle = 0
        assert self.accelerating_rotating_fly.angle == upwind_angle

    def test_step_should_rotate_fly(self):
        self.accelerating_rotating_fly.reset()
        self.accelerating_rotating_fly.step()
        rotated_angle = 0.5 * np.pi
        assert self.accelerating_rotating_fly.angle == pytest.approx(rotated_angle)


    def test_1st_step_to_right(self):
        self.accelerating_rotating_fly.reset()
        self.accelerating_rotating_fly.step()
        first_new_position = np.array([0.01, 0])
        assert np.array_equal(self.accelerating_rotating_fly.position, first_new_position)
    '''
    def test_2nd_step_position(self):
        self.accelerating_rotating_fly.reset()
        self.accelerating_rotating_fly.step()
        self.accelerating_rotating_fly.step()
        #First step to right
        t1_step = np.array([0.01, 0])
        #Second step up, twice as far due to acceleration
        t2_step = 2 * np.array([0, 0.01])
        t2_pos = np.array([0, 0]) + t1_step + t2_step
        assert np.array_equal(self.accelerating_rotating_fly.position, t2_pos)
    '''

