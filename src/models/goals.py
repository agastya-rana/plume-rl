from dataclasses import dataclass

import numpy as np
GOAL_RADIUS = 25
GOAL_Y = 450
GOAL_X = 150


@dataclass
class GoalZone:
    goal_x: int | float = GOAL_X
    goal_y: int | float = GOAL_Y
    goal_radius: int | float = GOAL_RADIUS

    def is_in_goal_zone(self, test_position: np.ndarray) -> bool:
        x_deviation = test_position[0] - self.goal_x
        y_deviation = test_position[1] - self.goal_y
        deviation = np.sqrt(x_deviation ** 2 + y_deviation ** 2)
        return deviation < self.goal_radius
