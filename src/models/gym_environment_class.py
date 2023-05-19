import numpy as np
from gym import Env
from gym.spaces import Box, Discrete

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

	def __init__(self, rng, config):


		self.fly_spatial_parameters = FlySpatialParameters(config)
		self.odor_features = OdorFeatures(config)
		self.odor_plume = OdorPlumeFromMovie(config)
		#self.reward_flag = RewardSchemeEnum.GOAL_ZONE

		#order = conc, grad, hrc, int, t_L_prev, t_L_current, theta

		self.observation_space = Box(low = np.array([0, -np.inf, -np.inf, 0, 0, 0, 0]), high = np.array([np.inf, np.inf, np.inf, 1, np.inf, np.inf, 2*np.pi]))
		self.obs_dim = config['OBSERVATION_DIMENSION']
		self.action_space = Discrete(config['NUM_ACTIONS'])

		self.goal_radius = config['GOAL_RADIUS_MM']
		self.source_location = config['SOURCE_LOCATION_MM']
		self.dt = config['DELTA_T_S']
		self.per_step_reward = config['PER_STEP_REWARD']
		self.rng = rng
		self.config = config
		self.max_frames = config['STOP_FRAME']
		self.episode_incrementer = 0
		self.flip_list = self.rng.choice(np.array([0,1]), size = config['N_EPISODES']).astype(bool)
		self.min_reset_x = config['MIN_RESET_X_MM']
		self.max_reset_x = config['MAX_RESET_X_MM']
		self.min_reset_y = config['MIN_RESET_Y_MM']
		self.max_reset_y = config['MAX_RESET_Y_MM']
		self.min_turn_dur = config['MIN_TURN_DUR_S']
		self.excess_turn_dur = config['EXCESS_TURN_DUR_S']
		self.theta_random_bounds = np.array([config['INIT_THETA_MIN'], config['INIT_THETA_MAX']])
		self.all_episode_rewards = np.zeros(config['N_EPISODES']) + np.nan

		self.initial_max_reset_x = self.min_reset_x + self.goal_radius

		self.x_random_bounds = np.array([self.min_reset_x, self.initial_max_reset_x])
		self.y_random_bounds = np.array([self.min_reset_y, self.max_reset_y])

		self.shift_episodes = config['SHIFT_EPISODES']


	def reset(self):

		self.total_episode_reward = 0
		self.odor_plume.reset(flip = self.flip_list[self.episode_incrementer], rng = self.rng)
		self.turn_durs = self.min_turn_dur + self.rng.exponential(scale = self.excess_turn_dur, size = self.max_frames)
		self.num_turns = 0
		odor_on = self.odor_plume.frame > self.config['CONCENTRATION_BASE_THRESHOLD']
		odor_on_indices = np.transpose(odor_on.nonzero())
		valid_locations = odor_on_indices*self.config['MM_PER_PX']

		if (self.episode_incrementer > 0) & (self.episode_incrementer % self.shift_episodes == 0):
			old_reset_x = max_reset_x
			old_reset_y = reset_y_radius
			max_reset_x = np.min([self.max_reset_x, 5*int(self.episode_incrementer/self.shift_episodes)*self.source_radius+self.initial_max_reset_x])
			self.x_random_bounds = np.array([self.min_reset_x, max_reset_x])

		self.fly_spatial_parameters.randomize_parameters(rng=self.rng, x_bounds=self.x_random_bounds, y_bounds=self.y_random_bounds, 
			theta_bounds = self.theta_random_bounds, valid_locations=valid_locations)

		self.odor_features.clear()
		odor_obs = self.odor_features.update(theta = self.fly_spatial_parameters.theta, 
			pos = self.fly_spatial_parameters.position, odor_frame = self.odor_plume.frame)
		all_obs = np.zeros(self.obs_dim)
		all_obs[0:-1] = odor_obs
		all_obs[-1] = self.fly_spatial_parameters.theta

		#self.episode_incrementer += 1
		

		return all_obs


	def step(self, action):

		self.odor_plume.advance(rng=self.rng)

		if action == 0 or action==3:

			self.fly_spatial_parameters.update_params(action)
			odor_obs = self.odor_features.update(theta = self.fly_spatial_parameters.theta, 
				pos = self.fly_spatial_parameters.position, odor_frame = self.odor_plume.frame)

			all_obs = np.zeros(self.obs_dim)
			all_obs[0:-1] = odor_obs
			all_obs[-1] = self.fly_spatial_parameters.theta

			reward = self.per_step_reward

			if self.odor_plume.frame_number >= self.max_frames:				
				done = True					

			else:
				done = False


		#turning actions 1 is left, 2 is right
		elif action == 1 or action == 2:

			turn_dur = self.turn_durs[self.num_turns]
			num_steps = int(turn_dur/self.dt)
			reward = 0

			for i in range(0,num_steps):

				self.odor_plume.advance(rng = self.rng)
				#print('in turn frame number = ', self.odor_plume.frame_number)
				self.fly_spatial_parameters.update_params(action)
				odor_obs = self.odor_features.update(theta = self.fly_spatial_parameters.theta, 
					pos = self.fly_spatial_parameters.position, odor_frame = self.odor_plume.frame)
				reward += self.per_step_reward

				all_obs = np.zeros(self.obs_dim)
				all_obs[0:-1] = odor_obs
				all_obs[-1] = self.fly_spatial_parameters.theta				

				if self.odor_plume.frame_number >= self.max_frames:				
					done = True	
					break				

				else:
					done = False


			self.num_turns += 1


		else:

			raise ValueError("action not integer 0, 1, 2 or 3") 


		x = self.fly_spatial_parameters.position[0]
		y = self.fly_spatial_parameters.position[1]

		in_rad = (x-self.source_location[0])**2 + (y-self.source_location[1])**2 < self.goal_radius ** 2

		if in_rad:

			done = True
			reward = 1


		self.total_episode_reward += reward

		if done:

			self.all_episode_rewards[self.episode_incrementer] = self.total_episode_reward
			self.episode_incrementer += 1

		info = {'concentration':all_obs[0], 'gradient':all_obs[1], 'hrc':all_obs[2], 
		'intermittency':all_obs[3], 't_L_prev':all_obs[4], 't_L_current':all_obs[5], 'theta':all_obs[6]}

		return all_obs, reward, done, info















