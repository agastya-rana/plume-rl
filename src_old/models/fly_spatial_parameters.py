import numpy as np 

class FlySpatialParameters():

	def __init__(self, config):

		self.theta = 0
		self.position = np.array([0,0])
		self.turn_ang_spd = config['TURN_ANG_SPEED_RAD_PER_S']
		self.walk_spd = config['WALK_SPEED_MM_PER_S']
		self.dt = config['DELTA_T_S']
		self.walk_step_size = self.walk_spd*self.dt
		self.ang_step_size = self.turn_ang_spd*self.dt 


	def update_params(self, action):

		#go straight
		if action == 0:

			self.dx = self.walk_step_size*np.cos(self.theta)
			self.dy = self.walk_step_size*np.sin(self.theta)

			vec = np.array([self.dx, self.dy])

			self.position += vec

		#turn left-note that this will typically be enforced for several time-steps in the step method
		elif action == 1:

			self.theta += self.ang_step_size

			if self.theta > 2*np.pi:

				self.theta = self.theta %2*np.pi


		#turn right-note that this will typically be enforced for several time-steps in the step method
		elif action == 2:

			self.theta -= self.ang_step_size

			if self.theta < 0:

				self.theta = self.theta % 2*np.pi



		#note, not doing anything for stop
		elif action == 3:

			pass


		else:

			raise ValueError("action not integer 0, 1, 2 or 3") 


		#note, we don't need anything for stop: action = 3


	def randomize_parameters(self, rng, x_bounds, y_bounds, theta_bounds, valid_locations):

		if valid_locations is not None:
			x_too_high = valid_locations[:, 0] > x_bounds[1]
			x_too_low = valid_locations[:, 0] < x_bounds[0]
			y_too_high = valid_locations[:, 1] > y_bounds[1]
			y_too_low = valid_locations[:, 1] < y_bounds[0]
			keep_indices = ~x_too_low & ~x_too_high & ~y_too_low & ~y_too_high
			valid_locations = valid_locations[keep_indices, :]
			n_valid_locations = valid_locations.shape[0]
		    
			assert n_valid_locations > 0
			random_valid_index = rng.choice(n_valid_locations, 1)
			random_x = valid_locations[random_valid_index, 0].item()
			random_y = valid_locations[random_valid_index, 1].item()


		else:
			random_x = rng.uniform(low=x_bounds[0], high=x_bounds[1])
			random_y = rng.uniform(low=y_bounds[0], high=y_bounds[1])
        
		random_position = np.array([random_x, random_y]) 
		self.position = random_position
		self.theta = rng.uniform(low = theta_bounds[0], high = theta_bounds[1])





