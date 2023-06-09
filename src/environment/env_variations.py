## This file implements variations of the standard FlyNavigator environment in gym_environment_class.py
## The variations are:
## 1. An environment with larger timesteps, and potentially integration of odor features by convolving with some filter
## 2. An environment with explicit history stored in the state space; this can be used to train DQN with history dependence;

import numpy as np
import gym
from gym_environment_class import FlyNavigator
from gym.spaces import Box, Discrete, MultiDiscrete
import copy

class IntegratedTimestepsNavigator(FlyNavigator):
	def __init__(self, integrated_dt=0.1, features_filter_type='exp', features_filter_size=5, **kwargs):
		super().__init__(**kwargs)
		self.integrated_dt = integrated_dt
		self.advance_timesteps = int(self.integrated_dt/self.dt)
		self.features_filter_size = features_filter_size
		self.features_filter_type = features_filter_type
		self.filtered_obs = np.zeros((self.num_odor_obs,))
	
	def step(self, action):
	## Step the environment with the given action
	## Step method in gym takes an action and returns the next state, reward, done, and info
		self.prev_theta = copy.deepcopy(self.fly_spatial_parameters.theta)
		self.prev_conc = copy.deepcopy(self.all_obs[0]) ## assumes that the first observation is the concentration
		reward = 0
		## Deal with actions that don't involve turning
		if action == 0 or action == 3:
			for _ in range(self.advance_timesteps):
				self.odor_plume.advance(rng=self.rng)
				self.fly_spatial_parameters.update_params(action)
				self._update_state()
				reward += self.per_step_reward
				if self.odor_plume.frame_number == self.max_frames:	
					self.done = True
					break

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
				self.all_episode_success.append(1) if self.reached_source else self.all_episode_success.append(0)
			self.episode_incrementer += 1

		info = {}
		return self.all_obs, reward, self.done, info
	
	def _update_state(self):
		odor_obs = self.odor_features.update(theta = self.fly_spatial_parameters.theta, 
			pos = self.fly_spatial_parameters.position, odor_frame = self.odor_plume.frame) ## Update the odor features at initalized fly location
		self.filtered_obs += 1/self.advance_timesteps*(odor_obs - self.filtered_obs)
		self.all_obs[:self.num_odor_obs] = self.filtered_obs
		self._add_theta_observation()

class HistoryNavigator(FlyNavigator):
	def __init__(self, history_len=5, store_angles=False, **kwargs):
		super().__init__(**kwargs)
		self.history_len = history_len ## note that this doesn't include the current observation
		self.history = np.zeros((self.num_odor_obs, self.history_len))
		## all_obs initialization depends on whether we use sin cos or not
		self.all_obs = np.zeros((self.num_odor_obs*(self.history_len+1),))
		self.store_angles = store_angles ## If true, store the fly angle history too TODO: implement this
		## In the following, I am leaving self.observables (names of observed features) unchanged; hopefully this doesn't mess up any other functions that call this
		if self.discrete_obs:
			all_obs_inds = copy.deepcopy(self.odor_features.discretization_index)*(self.history_len+1)
			all_obs_inds.append(self.theta_discretization) #note that for discretized states it doesn't make sense to split into sin and cos so this assumes only 1 theta observable
			self.observation_space = MultiDiscrete(all_obs_inds)
		else:
			if self.use_cos_and_sin:
				self.observable_bounds = np.vstack((self.odor_features.feat_bounds,)*(self.history_len+1) +  (np.array([[-1, 1], [-1, 1]]),)) ## bounds for cos and sin theta
			else:
				self.observable_bounds = np.vstack((self.odor_features.feat_bounds,)*(self.history_len+1) +  (np.array([[0, 2*np.pi]]),)) ## bounds for theta
			self.observation_space = Box(low=self.observable_bounds[:, 0], high=self.observable_bounds[:, 1])
		
		self.theta_dim = 2 if self.use_cos_and_sin else 1
		self.obs_dim = self.num_odor_obs*(self.history_len+1)  ## 2 for cos and sin theta, 1 for theta
		## We use convention where most recent observation is first in the array (see _update_state)
		self.all_obs = np.zeros(self.obs_dim).astype(int) if self.discrete_obs else np.zeros(self.obs_dim).astype('float32') ## Initialize all observables to 0
	
	def step(self, action):
		## Step the environment with the given action
		## Add the current observation to the history, and remove the oldest observation
		self.history = np.roll(self.history, 1, axis=0)
		self.history[:,0] = self.all_obs[:self.num_odor_obs]
		super().step(action)
	
	def _update_state(self):
		odor_obs = self.odor_features.update(theta = self.fly_spatial_parameters.theta, 
			pos = self.fly_spatial_parameters.position, odor_frame = self.odor_plume.frame) ## Update the odor features at initalized fly location
		self.all_obs[:self.num_odor_obs] = odor_obs
		self.all_obs[self.num_odor_obs:-self.theta_dim] = self.history.flatten()
		self._add_theta_observation() ## remember, this adds theta to the last one or two frames of observation