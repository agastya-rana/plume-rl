#from dataclasses import dataclass

import numpy as np

#GOAL_RADIUS = 25
#SOURCE_LOCATION = np.array([150,450])
#GOAL_X = SOURCE_LOCATION[0]
#GOAL_Y = SOURCE_LOCATION[1]


#@dataclass
class GoalZone:

	def __init__(self, config):

		self.goal_x = config['SOURCE_LOCATION_MM'][0]
		self.goal_y = config['SOURCE_LOCATION_MM'][1]
		self.goal_radius = config['GOAL_RADIUS_MM']

	def is_in_goal_zone(self, test_position: np.ndarray):

		x_deviation = test_position[0] - self.goal_x
		y_deviation = test_position[1] - self.goal_y
		deviation = np.sqrt(x_deviation ** 2 + y_deviation ** 2)
    	
		return deviation < self.goal_radius
