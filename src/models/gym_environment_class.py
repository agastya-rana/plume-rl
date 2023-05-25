import numpy as np
from gym import Env
from gym.spaces import Box, Discrete, MultiDiscrete
import imageio
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

from src.models.fly_spatial_parameters import FlySpatialParameters
from src.models.odor_plumes import *
from src.models.odor_senses import *

class FlyNavigator(Env):

	"""
	This is structured like the OpenAI Gym Environments, with reset and step actions.
	When it is initialized, it takes a bunch of parameters that define how rewards are
	calculated, what the odor plume is, what the agent senses (odor features).
	Actual instances of this class are composed in motion_environment_factory.py
	This class specifies the reset and step actions that are needed for openAI gym.
	The 'render' method is part of Gym environments but isn't implemented yet.
    """
	
	metadata = {'render.modes': ['human']}
	## Define the bounds for each of the observables, update as superset of observables grows
	observable_bounds = {'conc': [0, np.inf], 'grad': [-np.inf, np.inf], 'hrc': [-np.inf, np.inf], 'int': [0, 1], 't_L_prev': [0, np.inf], 't_L_current': [0, np.inf], 'theta': [0, 2*np.pi]}

	def __init__(self, rng, config):
		## Initialize spatial parameters, odor features, and odor plume
		self.fly_spatial_parameters = FlySpatialParameters(config)
		odor_class = config['ODOR_FEATURES_CLASS']
		self.odor_features = odor_class(config)
		self.odor_plume = OdorPlumeFromMovie(config)

		if config['USE_COSINE_AND_SIN_THETA']:

			self.observables = copy.deepcopy(self.odor_features.odor_observables)
			self.observables.append('cos_theta')
			self.observables.append('sin_theta')
			odor_observable_bounds = self.odor_features.odor_observable_bounds
			num = np.shape(odor_observable_bounds)[0]
			self.observable_bounds = np.zeros((num+2, 2))
			self.observable_bounds[0:num,:] = odor_observable_bounds
			self.observable_bounds[num,0] = -1
			self.observable_bounds[num,1] = 1
			self.observable_bounds[num+1,0] = -1
			self.observable_bounds[num+1,1] = 1

		else:

			self.observables = copy.deepcopy(self.odor_features.odor_observables)
			self.observables.append('theta')
			odor_observable_bounds = self.odor_features.odor_observable_bounds
			num = np.shape(odor_observable_bounds)[0]
			self.observable_bounds = np.zeros((num+1, 2))
			self.observable_bounds[0:num,:] = odor_observable_bounds
			self.observable_bounds[num,0] = 0
			self.observable_bounds[num,1] = 2*np.pi

		if config['DISCRETIZE_OBSERVABLES']:

			assert self.odor_features.can_discretize, "Class does not have discretization capability"

		if config['USE_COSINE_AND_SIN_THETA']:

			assert not config['DISCRETIZE_OBSERVABLES'], "using sin and cos but trying to discretize-use theta directly instead"

		if config['DISCRETIZE_OBSERVABLES']:

			self.theta_discretization = config['THETA_DISCRETIZATION']
			all_obs_inds = copy.deepcopy(self.odor_features.discretization_index)
			all_obs_inds.append(self.theta_discretization) #note that for discretized states it doesn't make sense to split into sin and cos so this assumes only 1 theta observable
			self.observation_space = MultiDiscrete(all_obs_inds)
			self.theta_bins = np.linspace(0,2*np.pi,self.theta_discretization+1)

		else:

			self.observation_space = Box(low = np.array([self.observable_bounds[i][0] for i in range(len(self.observables))]), high = np.array([self.observable_bounds[i][1] for i in range(len(self.observables))]))
		
		self.use_cos_and_sin = config['USE_COSINE_AND_SIN_THETA']
		self.discretize_observables = config['DISCRETIZE_OBSERVABLES']
		self.num_odor_obs = len(self.odor_features.odor_observables)
		self.obs_dim = len(self.observables) 
		self.action_space = Discrete(config['NUM_ACTIONS'])
		self.goal_radius = config['GOAL_RADIUS_MM']
		self.source_location = config['SOURCE_LOCATION_MM']
		self.dt = config['DELTA_T_S']
		self.per_step_reward = config['PER_STEP_REWARD']
		self.rng = rng
		self.config = config
		self.max_frames = config['STOP_FRAME']
		self.episode_incrementer = 0
		self.min_reset_x = config['MIN_RESET_X_MM']
		self.max_reset_x = config['MAX_RESET_X_MM']
		self.min_reset_y = config['MIN_RESET_Y_MM']
		self.max_reset_y = config['MAX_RESET_Y_MM'] ## These are the bounds for the initial position of the fly
		self.min_turn_dur = config['MIN_TURN_DUR_S'] ## Minimum turn duration in seconds
		self.excess_turn_dur = config['EXCESS_TURN_DUR_S'] ## Scale parameter for the exponential distribution of turn durations
		self.theta_random_bounds = np.array([config['INIT_THETA_MIN'], config['INIT_THETA_MAX']])
		self.all_episode_rewards = []
		self.all_episode_success = []
		self.source_reward = config['SOURCE_REWARD']

		self.initial_max_reset_x = config['INITIAL_MAX_RESET_X_MM']
		self.reset_x_shift = config['RESET_X_SHIFT_MM']

		self.x_random_bounds = np.array([self.min_reset_x, self.initial_max_reset_x])
		self.y_random_bounds = np.array([self.min_reset_y, self.max_reset_y])
		self.shift_episodes = config['SHIFT_EPISODES']
		self.trajectory_number = 0
		self.fly_trajectory = np.zeros((self.max_frames, 2)) + np.nan
		self.fig, self.ax = plt.subplots()
		self.video = config['RENDER_VIDEO'] #whether 
		self.writer = imageio.get_writer('movie.mp4', fps=30)

	def _add_theta_observation(self):

		if self.use_cos_and_sin:

			#again, doesn't make sense to use cos and sin if discretizing, so this assumes no discretization

			self.all_obs[-2] = np.cos(self.fly_spatial_parameters.theta)
			self.all_obs[-1] = np.sin(self.fly_spatial_parameters.theta)

		else:

			if self.discretize_observables:

				val = np.digitize(self.fly_spatial_parameters.theta, self.theta_bins)
				self.all_obs[-1] = val - 1 #-1 needed because digitize calls fist bin as bin 1 instead of bin 0

			else:

				self.all_obs[-1] = self.fly_spatial_parameters.theta

	def reset(self):
		## Reset method in gym returns the initial observation (state) of the environment
		self.total_episode_reward = 0
		flip = self.rng.choice(np.array([0,1])).astype(bool)
		self.odor_plume.reset(flip = flip, rng = self.rng)
		self.turn_durs = self.min_turn_dur + self.rng.exponential(scale = self.excess_turn_dur, size = self.max_frames)
		self.num_turns = 0
		odor_on = self.odor_plume.frame > self.config['CONCENTRATION_BASE_THRESHOLD']
		odor_on_indices = np.transpose(odor_on.nonzero())
		valid_locations = odor_on_indices*self.config['MM_PER_PX']

		if (self.episode_incrementer > 0) & (self.episode_incrementer % self.shift_episodes == 0):
			max_reset_x = np.min([self.max_reset_x, int(self.episode_incrementer/self.shift_episodes)*self.reset_x_shift+self.initial_max_reset_x])
			self.x_random_bounds = np.array([self.min_reset_x, max_reset_x])
		self.fly_spatial_parameters.randomize_parameters(rng=self.rng, x_bounds=self.x_random_bounds, y_bounds=self.y_random_bounds, 
			theta_bounds = self.theta_random_bounds, valid_locations=valid_locations) ## Randomize the fly's initial position and orientation

		self.odor_features.clear() ## Clear the odor features
		odor_obs = self.odor_features.update(theta = self.fly_spatial_parameters.theta, 
			pos = self.fly_spatial_parameters.position, odor_frame = self.odor_plume.frame) ## Update the odor features at initalized fly location
		self.all_obs = np.zeros(self.obs_dim)
		self.all_obs[0:self.num_odor_obs] = odor_obs
		self._add_theta_observation()

		if self.discretize_observables:

			self.all_obs = self.all_obs.astype(int)

		return self.all_obs


	def step(self, action):
		## Step method in gym takes an action and returns the next state, reward, done, and info
		self.odor_plume.advance(rng=self.rng)

		## Deal with actions that don't involve turning
		if action == 0 or action == 3:

			self.fly_spatial_parameters.update_params(action)
				
			odor_obs = self.odor_features.update(theta = self.fly_spatial_parameters.theta, 
				pos = self.fly_spatial_parameters.position, odor_frame = self.odor_plume.frame)

			self.all_obs = np.zeros(self.obs_dim)
			self.all_obs[0:self.num_odor_obs] = odor_obs
			#all_obs[-1] = self.fly_spatial_parameters.theta
			reward = self.per_step_reward
			if self.odor_plume.frame_number >= self.max_frames:		
				done = True
			else:
				done = False

		## Deal with actions that involve turning (1 is left, 2 is right)
		elif action == 1 or action == 2:
			## Turn duration is drawn from an exponential distribution with a minimum turn duration, with the samples drawn at reset
			turn_dur = self.turn_durs[self.num_turns]
			num_steps = int(turn_dur/self.dt)
			reward = 0
			## Note that by virtue of this 'turn time' being implemented in the step method, flies cannot learn information during a turn
			for i in range(0, num_steps):
				self.odor_plume.advance(rng = self.rng)
				#print('in turn frame number = ', self.odor_plume.frame_number)
				self.fly_spatial_parameters.update_params(action)
				odor_obs = self.odor_features.update(theta = self.fly_spatial_parameters.theta, 
					pos = self.fly_spatial_parameters.position, odor_frame = self.odor_plume.frame)
				reward += self.per_step_reward

				self.all_obs = np.zeros(self.obs_dim)
				self.all_obs[0:self.num_odor_obs] = odor_obs

				if self.odor_plume.frame_number >= self.max_frames:				
					done = True
					break				
				else:
					done = False
			self.num_turns += 1
		else:
			raise ValueError('Action must be 0, 1, 2, or 3')


		x = self.fly_spatial_parameters.position[0]
		y = self.fly_spatial_parameters.position[1]

		in_rad = (x-self.source_location[0])**2 + (y-self.source_location[1])**2 < self.goal_radius ** 2
		if in_rad:
			done = True
			reward = self.source_reward

		self.total_episode_reward += reward

		if done:

			self.all_episode_rewards.append(self.total_episode_reward)
			if reward == 1:
				self.all_episode_success.append(1)

			else:
				self.all_episode_success.append(0)

			self.episode_incrementer += 1

		self._add_theta_observation()

		if self.discretize_observables:

			self.all_obs = self.all_obs.astype(int)

		info = {}

		return self.all_obs, reward, done, info

	def draw_pointer(self, ax, position, angle, length=1.0, color='red'):
		# Calculate the vertices of the triangle
		x = position[0]
		y = position[1]
		vertices = np.array([[x - length/4, y], [x + length/4, y], [x, y + length]])

		# Rotate the vertices by the given angle around the center
		angle = angle - np.pi/2 # Correct for the fact that the triangle is drawn pointing up
		rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
		rotated_vertices = np.dot(vertices - position, rotation_matrix.T) + position
		# Plot the triangle
		ax.fill(rotated_vertices[:, 0], rotated_vertices[:, 1], color=color)

	def render(self, mode='human'):
		if mode == 'human':
			# Clear the previous plot
			self.ax.clear()
			# Plot odor background in grayscale
			self.ax.imshow(self.odor_plume.frame.T, cmap='gray', extent=(0, self.odor_plume.frame.shape[0]*self.config['MM_PER_PX'], 0, self.odor_plume.frame.shape[1]*self.config['MM_PER_PX']))
			# Plot the odor source
			self.ax.scatter(*self.source_location, color='green')
			## Plot the goal radius
			self.ax.add_patch(patches.Circle(self.source_location, self.goal_radius, color='green', fill=False))
			# Plot the current position and orientation of the fly
			self.draw_pointer(self.ax, self.fly_spatial_parameters.position, self.fly_spatial_parameters.theta, length=10,color='red')
			#self.ax.scatter(*self.fly_spatial_parameters.position, color='red')
			#self.ax.add_patch(patches.Arrow(*self.fly_spatial_parameters.position, np.cos(self.fly_spatial_parameters.theta), np.sin(self.fly_spatial_parameters.theta), width=0.5, head_width=4, color='red'))
			# Plot the trajectory of the fly
			self.fly_trajectory[self.trajectory_number] = self.fly_spatial_parameters.position
			self.trajectory_number += 1
			self.ax.plot(*zip(*self.fly_trajectory), color='cyan')

			# Set the plot limits
			self.ax.set_xlim(0, self.odor_plume.frame.shape[0]*self.config['MM_PER_PX'])
			self.ax.set_ylim(0, self.odor_plume.frame.shape[1]*self.config['MM_PER_PX'])
			
			if self.video:
				# Save current frame to the video file
				self.fig.canvas.draw() # draw the canvas, cache the renderer
				image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
				image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
				self.writer.append_data(image)

			# Annotate plot with current odor detections
			odor_features = self.odor_features.update(self.fly_spatial_parameters.theta, self.fly_spatial_parameters.position, self.odor_plume.frame)
			textstr = f"Concentration: {odor_features[0]:.2f}\nGradient: {odor_features[1]:.2f}\nVelocity: {odor_features[2]:.2f}"
			props = dict(boxstyle='round', facecolor='white', alpha=0.5)
			self.ax.text(0.05, 0.95, textstr, transform=self.ax.transAxes, fontsize=14,
				verticalalignment='top', bbox=props)
			
			# Draw the plot
			#plt.pause(0.0001)

		else:
			super(FlyNavigator, self).render(mode=mode)
		
		def close(self):
			if self.video:
				self.writer.close()
			super(FlyNavigator, self).close()










