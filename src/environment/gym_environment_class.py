import numpy as np
from gym import Env
from gym.spaces import Box, Discrete, MultiDiscrete
import imageio
import matplotlib
#matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from src.environment.fly_spatial_parameters import FlySpatialParameters
from src.environment.odor_plumes import *
from src.environment.odor_senses import *

class FlyNavigator(Env):
	"""
	An OpenAI Gym Environment for a fly navigating in an odor plume. Environment
	is defined by the step and rest methods. The step method takes an action and
	returns the next state, reward, and whether the episode is done. The reset method
	resets the environment to a random initial state.

	When it is initialized, it takes a bunch of parameters that define how rewards are
	calculated, what the odor plume is, what the agent senses (odor features).
    """
	
	def __init__(self, rng, config):
		state_dict = config["state"]
		agent_dict = config["agent"]
		plume_dict = config["plume"]
		training_dict = config["training"]
		self.gamma = training_dict["GAMMA"]

		## Initialize plume, odor features and fly parameters features
		if 'PLUME_DICT_LIST' in plume_dict:
			self.plume_dict_list = plume_dict['PLUME_DICT_LIST']
			self.plume_list = []
			for plume_config in plume_dict_list:
				if 'PLUME_TYPE' not in plume_config or plume_config['PLUME_TYPE'] == 'movie':
					make_plume_config = {'plume': plume_config} ## backwards compatibility
					self.plume_list.append(OdorPlumeFromMovie(make_plume_config))
				elif plume_config['PLUME_TYPE'] == 'ribbon':
					make_plume_config = {'plume': plume_config} ## just for backwards compatibility for now
					self.plume_list.append(StaticGaussianRibbon(make_plume_config))
			## Set plume probabilities; default is uniform
			self.plume_probs = plume_dict['PLUME_PROBS'] if 'PLUME_PROBS' in plume_dict else np.ones(len(self.plume_list))/len(self.plume_list)
			self.plume_inds = np.arange(len(self.plume_probs)) ## list of plume indices that gets sampled from with plume probs 
			self.all_plume_inds = []
			## Note that the following initializations init None values; only calling reset method will initialize these; it seems like we should just do this for everything?
		else:
			self.plume_dict_list = None
			if 'PLUME_TYPE' not in plume_dict or plume_dict['PLUME_TYPE'] == 'movie':
				self.odor_plume = OdorPlumeFromMovie(config) ## Defines the odor plume the fly is navigating in.
			elif plume_dict['PLUME_TYPE'] == 'ribbon':
				self.odor_plume = StaticGaussianRibbon(config)
			else:
				raise ValueError("Plume type not recognized")

		self.odor_features = OdorFeatures(config) ## Defines the odor features the fly senses.
		self.fly_spatial_parameters = FlySpatialParameters(config) ## True (x,y,theta)
	
		## If only single plume movie, then initialize here since reset method will not call init_plume_variables
		if 'PLUME_DICT_LIST' not in plume_dict:
			self._init_plume_variables(plume_dict)

		## Define observation space
		self.use_cos_and_sin = state_dict['USE_COSINE_AND_SIN_THETA'] if 'USE_COSINE_AND_SIN_THETA' in state_dict else True
		self.discrete_obs = state_dict["DISCRETE_OBSERVABLES"] if "DISCRETE_OBSERVABLES" in state_dict else False
		if self.discrete_obs:
			assert "THETA_DISCRETIZATION" in state_dict, "discrete observables but no theta discretization provided"
			self.theta_discretization = state_dict['THETA_DISCRETIZATION']
			all_obs_inds = copy.deepcopy(self.odor_features.discretization_index)
			all_obs_inds.append(self.theta_discretization) #note that for discretized states it doesn't make sense to split into sin and cos so this assumes only 1 theta observable
			self.observation_space = MultiDiscrete(all_obs_inds)
			self.theta_bins = np.linspace(0, 2*np.pi, self.theta_discretization+1)
			self.observables = copy.deepcopy(state_dict['FEATURES'])
			self.observables.append('theta')
		else:
			if self.use_cos_and_sin:
				assert not self.discrete_obs, "using sin and cos but trying to discretize; use theta directly instead"
				self.observable_bounds = np.vstack((self.odor_features.feat_bounds, np.array([[-1, 1], [-1, 1]]))) ## bounds for cos and sin theta
				self.observables = copy.deepcopy(state_dict['FEATURES'])
				self.observables.append('cos_theta')
				self.observables.append('sin_theta')
			else:
				self.observables = copy.deepcopy(state_dict['FEATURES'])
				self.observables.append('theta')
				self.observable_bounds = np.vstack((self.odor_features.feat_bounds, np.array([[0, 2*np.pi]]))) ## bounds for theta
			self.observation_space = Box(low=self.observable_bounds[:, 0], high=self.observable_bounds[:, 1])
		
		## Set observation arrays and action space
		self.num_odor_obs = len(state_dict['FEATURES'])
		self.obs_dim = len(self.observables)
		self.all_obs = np.zeros(self.obs_dim).astype(int) if self.discrete_obs else np.zeros(self.obs_dim).astype('float32') ## Initialize all observables to 0
		self.action_space = Discrete(4) ## 0: forward, 1: left, 2: right, 3: stop
		
		## Define agent parameters
		self.goal_radius = agent_dict['GOAL_RADIUS_MM']
		self.min_turn_dur = agent_dict['MIN_TURN_DUR_S'] ## Minimum turn duration in seconds
		self.excess_turn_dur = agent_dict['EXCESS_TURN_DUR_S'] ## Scale parameter for the exponential distribution of turn durations
		self.antenna_height = agent_dict['ANTENNA_LENGTH_MM']
		self.antenna_width = agent_dict['ANTENNA_WIDTH_MM']
		self.detection_threshold = state_dict['DETECTION_THRESHOLD'] if 'DETECTION_THRESHOLD' in state_dict else None
		self.detection_threshold_type = state_dict['DETECTION_THRESHOLD_TYPE'] if 'DETECTION_THRESHOLD_TYPE' in state_dict else "fixed"
				
		## Render parameters
		self.fig, self.ax = plt.subplots()
		self.video = training_dict['RENDER_VIDEO'] if 'RENDER_VIDEO' in training_dict else False ## Whether or not to render a video of the fly's trajectory
		self.writer = imageio.get_writer(os.path.join(training_dict['SAVE_DIRECTORY'], training_dict['MODEL_NAME']+".mp4"), fps=60) if self.video else None

		## Reward shaping parameters
		reward_dict = config['reward']
		self.source_reward = reward_dict['SOURCE_REWARD'] if 'SOURCE_REWARD' in reward_dict else 0
		self.per_step_reward = reward_dict['PER_STEP_REWARD'] if 'PER_STEP_REWARD' in reward_dict else 0
		self.conc_upwind_reward = reward_dict['CONC_UPWIND_REWARD'] if 'CONC_UPWIND_REWARD' in reward_dict else 0
		self.upwind_reward = reward_dict['UPWIND_REWARD'] if 'UPWIND_REWARD' in reward_dict else 0
		self.conc_reward = reward_dict['CONC_REWARD'] if 'CONC_REWARD' in reward_dict else 0
		self.radial_reward = reward_dict['RADIAL_REWARD'] if 'RADIAL_REWARD' in reward_dict else 0
		self.radial_conc_gating = reward_dict['RADIAL_CONC_GATING'] if 'RADIAL_CONC_GATING' in reward_dict else 0
		self.stray_reward = reward_dict['STRAY_REWARD'] if 'STRAY_REWARD' in reward_dict else 0
		self.wall_penalty = reward_dict['WALL_PENALTY'] if 'WALL_PENALTY' in reward_dict else 0
		
		## Misc
		self.rng = rng
		self.record_success = training_dict['RECORD_SUCCESS'] if 'RECORD_SUCCESS' in training_dict else False
		if self.record_success:
			self.all_episode_rewards = []
			self.all_episode_success = []
		self.episode_incrementer = 0
		self.trajectory_number = 0
		self.fly_trajectory = np.zeros((self.max_frames, 2)) + np.nan

	def _init_plume_variables(self, current_plume_dict):
		
		dt = current_plume_dict['DELTA_T_S'] if 'DELTA_T_S' in current_plume_dict else raise Exception("DELTA_T_S not in specified for plume {i}".format(i=self.plume_ind))
		self.dt = dt
		self.odor_features.dt = dt
		mm_per_px = current_plume_dict['MM_PER_PX'] if 'MM_PER_PX' in current_plume_dict else raise Exception("MM_PER_PX not in specified for plume {i}".format(i=self.plume_ind))
		self.mm_per_px = mm_per_px
		self.odor_features.mm_per_px = mm_per_px

		self.odor_features.max_t_L = self.dt*(current_plume_dict['STOP_FRAME']-current_plume_dict['MIN_FRAME'])

		## Set up odor features box
		self.odor_features.std_left_box, self.odor_features.std_right_box = self.odor_features.make_L_R_std_box(self.mm_per_px, self.antenna_height, self.antenna_width)
		self.odor_features.num_pts = np.shape(self.odor_features.std_left_box)[0]

		## Set episode termination features
		self.max_frames = current_plume_dict['STOP_FRAME'] if 'STOP_FRAME' in current_plume_dict else raise Exception("STOP_FRAME not in specified for plume {i}".format(i=self.plume_ind))
		self.source_location = plume_dict['SOURCE_LOCATION_MM'] if 'SOURCE_LOCATION_MM' in plume_dict else raise Exception("SOURCE_LOCATION_MM not in specified for plume {i}".format(i=self.plume_ind))
		
		## Define reset parameters
		(self.min_reset_x, self.max_reset_x), (self.min_reset_y, self.max_reset_y) = plume_dict["RESET_BOX_MM"] if 'RESET_BOX_MM' in plume_dict else raise Exception("RESET_BOX_MM not in specified for plume {i}".format(i=self.plume_ind))
		self.theta_random_bounds = np.array(plume_dict['INIT_THETA_BOUNDS']) if 'INIT_THETA_BOUNDS' in plume_dict else np.array([0, 2*np.pi])
		self.initial_max_reset_x = plume_dict['INITIAL_MAX_RESET_X_MM'] if 'INITIAL_MAX_RESET_X_MM' in plume_dict else self.max_reset_x
		self.reset_x_shift = plume_dict['RESET_X_SHIFT_MM'] if 'RESET_X_SHIFT_MM' in plume_dict else 0
		self.x_random_bounds = np.array([self.min_reset_x, self.initial_max_reset_x])
		self.y_random_bounds = np.array([self.min_reset_y, self.max_reset_y])
		self.shift_episodes = plume_dict['SHIFT_EPISODES'] if 'SHIFT_EPISODES' in plume_dict and self.reset_x_shift > 0 else 0
		
		if 'WALL_BOX_MM' in plume_dict:
			(self.min_wall_x, self.max_wall_x), (self.min_wall_y, self.max_wall_y) = plume_dict["WALL_BOX_MM"]
		elif self.wall_penalty > 0:
			raise Exception("WALL_BOX_MM not in specified for plume {i}".format(i=self.plume_ind))

		## Set up fly spatial parameters
		self.fly_spatial_parameters.dt = current_plume_dict['DELTA_T_S']
		self.fly_spatial_parameters.walk_step_size = self.fly_spatial_parameters.dt*self.fly_spatial_parameters.walk_spd
		self.fly_spatial_parameters.ang_step_size = self.fly_spatial_parameters.dt*self.fly_spatial_parameters.turn_ang_spd		

	def reset(self):
		## Reset method in gym returns the initial observation (state) of the environment
		self.total_episode_reward = 0
		self.reached_source = False
		self.done = False

		## Reset render variables
		self.trajectory_number = 0
		self.fly_trajectory = np.zeros((self.max_frames, 2)) + np.nan - hopefully this still works by initing in reset

		if self.plume_dict_list is not None:
			self.plume_ind = self.rng.choice(self.plume_inds, p = self.plume_probs)
			self.all_plume_inds.append(self.plume_ind)
			self.odor_plume = self.plume_list[self.plume_ind]
			current_plume_dict = self.plume_dict_list[self.plume_ind]
			self._init_plume_variables(current_plume_dict)
		
		## Reset odor plumes and turning distributions
		flip = self.rng.choice([True, False])
		self.odor_plume.reset(flip = flip, rng = self.rng)
		self.turn_durs = self.min_turn_dur + self.rng.exponential(scale = self.excess_turn_dur, size = self.max_frames)
		self.num_turns = 0

		## Initialize fly position and orientation
		odor_on = self.odor_plume.frame > self.detection_threshold
		odor_on_indices = np.transpose(odor_on.nonzero())
		valid_locations = odor_on_indices*self.mm_per_px

		if (self.episode_incrementer > 0) & (self.episode_incrementer % self.shift_episodes == 0):
			max_reset_x = np.min([self.max_reset_x, int(self.episode_incrementer/self.shift_episodes)*self.reset_x_shift+self.initial_max_reset_x])
			self.x_random_bounds = np.array([self.min_reset_x, max_reset_x])
		self.fly_spatial_parameters.randomize_parameters(rng=self.rng, x_bounds=self.x_random_bounds, y_bounds=self.y_random_bounds, 
			theta_bounds = self.theta_random_bounds, valid_locations=valid_locations) ## Randomize the fly's initial position and orientation
		
		## Clear the odor features
		self.odor_features.clear() ## Clear the odor features
		self._update_state() ## Update the state
		self.previous_distance = np.linalg.norm(self.fly_spatial_parameters.position-self.source_location)
		return self.all_obs

	def _update_state(self):
		## Update the state with odor features and theta
		odor_obs = self.odor_features.update(theta = self.fly_spatial_parameters.theta, 
			pos = self.fly_spatial_parameters.position, odor_frame = self.odor_plume.frame) ## Update the odor features at initalized fly location
		self.all_obs[:self.num_odor_obs] = odor_obs
		self._add_theta_observation()
	
	def _add_theta_observation(self):
		## Add theta to observation
		if self.use_cos_and_sin:
			#again, doesn't make sense to use cos and sin if discretizing, so this assumes no discretization
			self.all_obs[-2] = np.cos(self.fly_spatial_parameters.theta)
			self.all_obs[-1] = np.sin(self.fly_spatial_parameters.theta)
		else:
			if self.discrete_obs:
				val = np.digitize(self.fly_spatial_parameters.theta, self.theta_bins)
				self.all_obs[-1] = val - 1 ## -1 needed because digitize calls fist bin as bin 1 instead of bin 0
			else:
				self.all_obs[-1] = self.fly_spatial_parameters.theta

	def step(self, action):
		## Step method in gym takes an action and returns the next state, reward, done, and info
		## Store the following for shaping:
		self.prev_theta = copy.deepcopy(self.fly_spatial_parameters.theta)
		self.prev_conc = copy.deepcopy(self.all_obs[0]) ## assumes that the first observation is the concentration
		## Deal with actions that don't involve turning
		if action == 0 or action == 3:
			self.odor_plume.advance(rng=self.rng)
			self.fly_spatial_parameters.update_params(action)
			self._update_state()
			reward = self.per_step_reward
			if self.odor_plume.frame_number == self.max_frames:	
				self.done = True
		
		## Deal with actions that involve turning (1 is left, 2 is right)
		elif action == 1 or action == 2:
			## Turn duration is drawn from an exponential distribution with a minimum turn duration, with the samples drawn at reset
			turn_dur = self.turn_durs[self.num_turns]
			num_steps = int(turn_dur/self.dt)
			reward = 0
			## Note that by virtue of this 'turn time' being implemented in the step method, flies cannot learn information during a turn
			for _ in range(num_steps):
				self.odor_plume.advance(rng = self.rng)
				#print('in turn frame number = ', self.odor_plume.frame_number)
				self.fly_spatial_parameters.update_params(action)
				reward += self.per_step_reward
				self._update_state()
				if self.odor_plume.frame_number == self.max_frames:
					self.done = True
					break
			self.num_turns += 1
		else:
			raise ValueError('Action must be 0, 1, 2, or 3')

		additional_rewards = self._get_additional_rewards()
		reward += additional_rewards
		self.total_episode_reward += reward

		if self.done:
			if self.record_success:
				self.all_episode_rewards.append(self.total_episode_reward)
				self.all_episode_success.append(1 if self.reached_source else 0)
			self.episode_incrementer += 1

		info = {"success": 1 if (self.done and self.reached_source) else 0, "total_episode_reward": self.total_episode_reward,}
		return self.all_obs, reward, self.done, info
	
	def _get_additional_rewards(self):
		## Get the additional rewards for the current step
		# Get the current distance from the source
		reward = 0
		pos = self.fly_spatial_parameters.position
		current_distance = np.linalg.norm(pos - self.source_location)
		if current_distance < self.goal_radius:
			self.done = True
			self.reached_source = True
			reward = self.source_reward
			return reward
		if self.wall_penalty: #for giving penalty for hitting walls
			outside = (pos[0] > self.wall_max_x) + (pos[0] < self.wall_min_x) + (pos[1] > self.wall_max_y) + (pos[1] < self.wall_min_y) #checking if out of bounds
			if outside:
				reward = self.wall_penalty
				self.done = True
				return reward

		if self.radial_reward: #for giving reward for decreasing distance from source
			if self.radial_conc_gating: ## change this for another parameter
				non_zero_check = self.all_obs[:self.num_odor_obs] != 0 #want to give this reward only when at least one odor feature is non-zero
				non_zero_check = np.sum(non_zero_check)>0
			else:
				non_zero_check = 1
			reward += self.radial_reward*(self.previous_distance-current_distance)*non_zero_check
			self.previous_distance = copy.deepcopy(current_distance)

		## Potential shaping rewards
		if self.conc_upwind_reward:
			new_potential = -self.conc_upwind_reward*self.all_obs[0]*np.cos(self.fly_spatial_parameters.theta)
			old_potential = -self.conc_upwind_reward*self.prev_conc*np.cos(self.prev_theta)
			reward += self.gamma*new_potential - old_potential

		if self.upwind_reward:
			non_zero_check = (np.sum(self.all_obs[:self.num_odor_obs] != 0) >0)
			new_potential = -self.upwind_reward*non_zero_check*np.cos(self.fly_spatial_parameters.theta)
			old_potential = -self.upwind_reward*non_zero_check*np.cos(self.prev_theta)
			reward += self.gamma*new_potential - old_potential
			
		if self.conc_reward:
			new_potential = self.conc_reward*self.all_obs[0]
			old_potential = self.conc_reward*self.prev_conc
			reward += self.gamma*new_potential - old_potential
		
		## Implementation of negative reward for straying far from nearest odor location on plume
		if self.stray_reward:
			nearest_odor_loc = self.odor_plume.nearest_odor_location(pos)
			nearest_odor_dist = np.linalg.norm(pos - nearest_odor_loc)
			reward += self.stray_reward*nearest_odor_dist
	
		return reward

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
	
		# Clear the previous plot
		self.ax.clear()
		# Plot odor background in grayscale
		self.ax.imshow(self.odor_plume.frame.T, cmap='gray', extent=(0, self.odor_plume.frame.shape[0]*self.mm_per_px, 0, self.odor_plume.frame.shape[1]*self.mm_per_px))
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
		self.ax.set_xlim(0, self.odor_plume.frame.shape[0]*self.mm_per_px)
		self.ax.set_ylim(0, self.odor_plume.frame.shape[1]*self.mm_per_px)
		
		if mode == 'human':
			if self.video:
				# Save current frame to the video file
				self.fig.canvas.draw() # draw the canvas, cache the renderer
				image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
				image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
				self.writer.append_data(image)

			# Annotate plot with current odor detections
			odor_features = self.odor_features.update(self.fly_spatial_parameters.theta, self.fly_spatial_parameters.position, self.odor_plume.frame)
			textstr = f"Concentration: {odor_features[0]:.2f}\nGradient: {odor_features[1]:.2f}"
			props = dict(boxstyle='round', facecolor='white', alpha=0.5)
			self.ax.text(0.05, 0.95, textstr, transform=self.ax.transAxes, fontsize=14,
				verticalalignment='top', bbox=props)
			
			# Draw the plot
			#plt.pause(0.0001)
		
		elif mode == 'rgb_array':
			self.fig.canvas.draw()
			image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
			image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
			return image

		else:
			super(FlyNavigator, self).render(mode=mode)
	
	def close(self):
		if self.video:
			self.writer.close()
		super(FlyNavigator, self).close()
