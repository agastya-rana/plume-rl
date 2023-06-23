## This file implements variations of the standard FlyNavigator environment in gym_environment_class.py
## The variations are:
## 1. An environment with larger timesteps, and potentially integration of odor features by convolving with some filter
## 2. An environment with explicit history stored in the state space; this can be used to train DQN with history dependence;

import numpy as np
import gym
from src.environment.gym_environment_class import FlyNavigator
from gym.spaces import Box, Discrete, MultiDiscrete
import copy

class IntegratedTimestepsNavigator(FlyNavigator):
	def __init__(self, rng, config):
		# integrated_dt=0.1, features_filter_type='exp', features_filter_size=5,
		super().__init__(rng, config)
		self.integrated_dt = config["agent"]["INTEGRATED_DT"]
		self.features_filter_size = config["agent"]["FEATURES_FILTER_SIZE"]
		self.advance_timesteps = int(self.integrated_dt/self.dt) ## This should be an exact multiple
		assert self.advance_timesteps*self.dt == self.integrated_dt
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
		self.filtered_obs += 1/self.features_filter_size*(odor_obs - self.filtered_obs)
		self.all_obs[:self.num_odor_obs] = self.filtered_obs
		self._add_theta_observation()

class HistoryNavigator(FlyNavigator):
	def __init__(self, rng, config):
		super().__init__(rng, config)
		self.history_len = config["state"]["HIST_LEN"] ## note that this doesn't include the current observation
		self.history = np.zeros((self.history_len, self.num_odor_obs))
		## all_obs initialization depends on whether we use sin cos or not
		self.store_angles = False ## If true, store the fly angle history too TODO: implement this
		## In the following, I am leaving self.observables (names of observed features) unchanged; hopefully this doesn't mess up any other functions that call this
		self.theta_dim = 2 if self.use_cos_and_sin else 1
		self.obs_dim = self.num_odor_obs*(self.history_len+1) + self.theta_dim ## 2 for cos and sin theta, 1 for theta
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
		## We use convention where most recent observation is first in the array (see _update_state)
		self.all_obs = np.zeros(self.obs_dim).astype(int) if self.discrete_obs else np.zeros(self.obs_dim).astype('float32') ## Initialize all observables to 0

	def step(self, action):
		## Step the environment with the given action
		## Add the current observation to the history, and remove the oldest observation
		self.history = np.roll(self.history, 1, axis=0)
		self.history[0, :] = self.all_obs[:self.num_odor_obs]
		return super().step(action)
	
	def _update_state(self):
		odor_obs = self.odor_features.update(theta = self.fly_spatial_parameters.theta, 
			pos = self.fly_spatial_parameters.position, odor_frame = self.odor_plume.frame) ## Update the odor features at initalized fly location
		self.all_obs[:self.num_odor_obs] = odor_obs
		self.all_obs[self.num_odor_obs:-self.theta_dim] = self.history.flatten()
		self._add_theta_observation() ## remember, this amends theta at the last one or two frames of observation

class HistoryTimestepNavigator(IntegratedTimestepsNavigator):
	def __init__(self, rng, config):
		super().__init__(rng, config)
		self.history_len = config["state"]["HIST_LEN"] ## note that this doesn't include the current observation
		self.history = np.zeros((self.history_len, self.num_odor_obs))
		## all_obs initialization depends on whether we use sin cos or not
		self.store_angles = False ## If true, store the fly angle history too TODO: implement this
		## In the following, I am leaving self.observables (names of observed features) unchanged; hopefully this doesn't mess up any other functions that call this
		self.theta_dim = 2 if self.use_cos_and_sin else 1
		self.obs_dim = self.num_odor_obs*(self.history_len+1) + self.theta_dim ## 2 for cos and sin theta, 1 for theta
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
		## We use convention where most recent observation is first in the array (see _update_state)
		self.all_obs = np.zeros(self.obs_dim).astype(int) if self.discrete_obs else np.zeros(self.obs_dim).astype('float32') ## Initialize all observables to 0
	
	def step(self, action):
		## Step the environment with the given action
		## Add the current observation to the history, and remove the oldest observation
		self.history = np.roll(self.history, 1)
		self.history[0, :] = self.all_obs[:self.num_odor_obs]
		return super().step(action)
	
	def _update_state(self):
		odor_obs = self.odor_features.update(theta = self.fly_spatial_parameters.theta, pos = self.fly_spatial_parameters.position, odor_frame = self.odor_plume.frame)
		self.filtered_obs += 1/self.features_filter_size*(odor_obs - self.filtered_obs)
		self.all_obs[:self.num_odor_obs] = self.filtered_obs
		self.all_obs[self.num_odor_obs:-self.theta_dim] = self.history.flatten()
		self._add_theta_observation()


class GoalDirectedNavigator(FlyNavigator):

	## Not yet integrated with history or timestep, but this should be doable easily
	## This class implements a navigator with a 2D box action space, interpreted as a vector whose direction is the desired direction of movement
	## Its magnitude may be interpreted as a certainty in direction of movement if desired
	## From this vector (the action), the step function decides whether to turn left, right (as in the original fly algorithm), or go straight (if the action is close enough to the current heading)

	def __init__(self, rng, config):
		super().__init__(rng, config)
		self.action_space = Box(low=-1, high=1, shape=(2,))
		goal_params = config["agent"]["GOAL_PARAMS"]
		self.certainty = goal_params["CERTAINTY"] ## 0 if agent is precise, otherwise this is the constant use to scale random decisions; good val is 0.3
		self.straight_tol = goal_params["STRAIGHT_TOL"] ## Tolerance for theta difference between goal and current direction; good val is probably 0.3 (radians)
		self.goal_access = goal_params["GOAL_ACCESS"] ## Whether the agent has access to the goal direction
		state_dict = config["state"]
		if self.goal_access:
			## Need to now change the observation space and related stuff to account for this
			if self.discrete_obs:
				assert self.certainty == 0, "Goal direction has no magnitude for dicrete agent"
				assert self.odor_features.can_discretize, "Set of features used does not have discretization capability"
				self.theta_discretization = state_dict['THETA_DISCRETIZATION']
				all_obs_inds = copy.deepcopy(self.odor_features.discretization_index)
				all_obs_inds.append(self.theta_discretization) #note that for discretized states it doesn't make sense to split into sin and cos so this assumes only 1 theta observable
				all_obs_inds.append(self.theta_discretization) # this is the goal theta				
				self.observation_space = MultiDiscrete(all_obs_inds)
				self.theta_bins = np.linspace(0, 2*np.pi, self.theta_discretization+1)
				self.observables = copy.deepcopy(state_dict['FEATURES'])
				self.observables.append('goal direction')				
				self.observables.append('theta')
			else:
				if self.use_cos_and_sin:
					assert not self.discrete_obs, "using sin and cos but trying to discretize; use theta directly instead"
					self.observable_bounds = np.vstack((self.odor_features.feat_bounds, np.array([[-1, 1]]*4))) ## bounds for cos and sin theta
					self.observables = copy.deepcopy(state_dict['FEATURES'])
					self.observables.append('goal_cos_theta')
					self.observables.append('goal_sin_theta')
					self.observables.append('cos_theta')
					self.observables.append('sin_theta')

				else:
					self.observables = copy.deepcopy(state_dict['FEATURES'])
					self.observables.append('goal_theta')
					self.observables.append('theta')
					self.observable_bounds = np.vstack((self.odor_features.feat_bounds, np.array([[0, 2*np.pi]]*2))) ## bounds for theta
				self.observation_space = Box(low=self.observable_bounds[:, 0], high=self.observable_bounds[:, 1])
			
			self.num_odor_obs = len(state_dict['FEATURES'])
			self.obs_dim = len(self.observables)
			self.all_obs = np.zeros(self.obs_dim).astype(int) if self.discrete_obs else np.zeros(self.obs_dim).astype('float32') ## Initialize all observables to 0

	def step(self, action):
		## Find goal direction and magnitude
		goal_mag = np.linalg.norm(action)
		goal_vec = action/goal_mag
		goal_theta = np.arctan2(goal_vec[1], goal_vec[0])
		self._add_goal_obs(goal_theta)
		## Find the difference between the goal direction and the current direction
		theta_diff = goal_theta - self.fly_spatial_parameters.theta
		## Wrap the difference to be between -pi and pi
		theta_diff = (theta_diff + np.pi) % (2*np.pi) - np.pi
		## If certainty is non-zero, then randomize theta_diff as normal with std deviation proportional to inverse of goal_mag
		if self.certainty:
			theta_diff += self.rng.normal(loc=0, scale=self.certainty/goal_mag)
		## If the difference is small enough, go straight; if positive, turn left; if negative, turn right
		if np.abs(theta_diff) < self.straight_tol:
			action_new = 0
		elif theta_diff > 0:
			action_new = 1
		else:
			action_new = 2
		return super().step(action_new) ## Note that the _update_state function called has dynamic binding, so it will call the one in this class
	
	def _add_goal_obs(self, goal_action):
		if self.use_cos_and_sin:
			#again, doesn't make sense to use cos and sin if discretizing, so this assumes no discretization
			self.all_obs[-4] = goal_action[0]
			self.all_obs[-3] = goal_action[1]
		else:
			goal_theta = np.arctan2(goal_action[1], goal_action[0])
			if self.discrete_obs:
				val = np.digitize(goal_theta, self.theta_bins)
				self.all_obs[-2] = val - 1 ## -1 needed because digitize calls fist bin as bin 1 instead of bin 0
			else:
				self.all_obs[-2] = goal_theta