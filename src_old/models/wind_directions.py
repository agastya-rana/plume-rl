import numpy as np

from src.models.geometry import AngleField


class WindDirections:
    wind_angle: float = AngleField()
    crosswind_a: float = AngleField()
    crosswind_b: float = AngleField()
    upwind: float = AngleField()

    def __init__(self, wind_angle: float = 0):
        self.wind_angle = wind_angle
        self.crosswind_a = wind_angle + (np.pi / 2)
        self.crosswind_b = wind_angle - (np.pi / 2)
        self.upwind = wind_angle + np.pi

    def __repr__(self):
        return 'WindDirections(wind = {0})'.format(self.wind_angle)
