from src.models.odor_senses import *
from src.models.gym_environment_class import FlyNavigator
#from stable_baselines3.deepq.policies import MlpPolicy
#from stable_baselines3 import DQN
#import stable_baselines3.common.utils

import numpy as np 
#from stable_baselines3 import DQN
import time
import sys 
import os

config_dict = {
	
	"NUM_ACTIONS": 4,
	"USE_COSINE_AND_SIN_THETA":False,
	"CONCENTRATION_BASE_THRESHOLD": 100, #this is the value that's good for movies. Do not change this to account for normalization-this happens internally.  
	"CONCENTRATION_THRESHOLD_STYLE": "fixed",
	"ODOR_FEATURES_CLASS": OdorFeatures_no_temporal,
	"DISCRETIZE_OBSERVABLES": True,
	"THETA_DISCRETIZATION": 6,
	"TEMPORAL_FILTER_TIMESCALE_S": 1,
	"TEMPORAL_THRESHOLD_ADAPTIVE_TIMESCALE_S":5,
	"TEMPORAL_FILTER_ALL":False,
	"MM_PER_PX": 0.2,
	"ANTENNA_LENGTH_MM": 0.41,
	"ANTENNA_WIDTH_MM": 0.21,
	"MAX_CONCENTRATION": 255,
	"NORMALIZE_ODOR_FEATURES": True,
	"WALK_SPEED_MM_PER_S": 10,
	"DELTA_T_S": 1/60,
	"PER_STEP_REWARD": 0,
	"SOURCE_REWARD": 100,
	"WITH_ORIENTATION": False,
	"USE_MOVIE": True,
	"MOVIE_PATH": None,
	"MIN_FRAME": 500,
	"STOP_FRAME": 600,
	"RESET_FRAME_RANGE": np.array([501,505]),
	"SOURCE_LOCATION_MM": np.array([30,90]),
	"GOAL_RADIUS_MM": 10, #success radius in mm
	"N_EPISODES" : 3, # How many independently initialized runs to train on
	"MAX_ALPHA": 0.1, # Learning rate
	"MIN_ALPHA": 0.001,
	"SOFTMAX": True,
	"GAMMA":0.95, # Reward temporal discount factor
	"MIN_EPSILON":0.01, # Asymptote of decaying exploration rate
	"MIN_RESET_X_MM": 40, # Initialization condition-minimum agent x in mm
	"INITIAL_MAX_RESET_X_MM": 45,
	"MAX_RESET_X_MM": 300, # Initialization condition-maximum agent x in mm
	"MIN_RESET_Y_MM": 0,
	"MAX_RESET_Y_MM": 180,
	"SOURCE_LOC_MM": np.array([30,90]), #Source location in mm
	"INIT_THETA_MIN": 0,
	"INIT_THETA_MAX": 2*np.pi,
	"TURN_ANG_SPEED_RAD_PER_S": 100*np.pi/180,
	"MIN_TURN_DUR_S": 0.18,
	"EXCESS_TURN_DUR_S": 0.18,
	"SHIFT_EPISODES": 100,
	"RESET_X_SHIFT_MM": 5,
	"RENDER_VIDEO": True,



}

seed = int(sys.argv[1])
rng = np.random.default_rng(seed)
plume_movie_path = os.path.join('..','src', 'data', 'plume_movies', 'intermittent_smoke.avi')
#plume_movie_path = '../src/data/plume_movies/intermittent_smoke.avi'

config_dict['MOVIE_PATH'] = plume_movie_path

environment = FlyNavigator(rng = rng, config = config_dict)

TEMPERATURE = config_dict['SOURCE_REWARD']/200

q_shape = np.append(environment.observation_space.nvec,environment.action_space.n)
q_table = np.zeros(shape=q_shape)
#print(q_shape)


fraction = 3/4
alpha_arr = np.zeros(config_dict['N_EPISODES'])
change_num = int(config_dict['N_EPISODES']*fraction)
inds = np.arange(0,change_num)

alpha_vals = config_dict['MAX_ALPHA'] - (config_dict['MAX_ALPHA']-config_dict['MIN_ALPHA'])/change_num * inds
alpha_arr[0:change_num] = alpha_vals
alpha_arr[change_num:] = config_dict['MIN_ALPHA'] 

#episode_incrementer = 0

all_Q_shape = np.append(q_shape, config_dict['N_EPISODES'])
all_Q = np.zeros(all_Q_shape)

for episode in range(0,config_dict['N_EPISODES']):

	obs = environment.reset()

	print(obs)
	print(environment.odor_features.concentration)

	done = False

	while not done:

		alpha = alpha_arr[episode]

		vals = q_table[tuple(obs)]

		print("vals = ", vals)

		probs = np.exp(vals/TEMPERATURE)/(np.sum(np.exp(vals/TEMPERATURE)))

		print('probs = ', probs)

		inds = np.arange(0,config_dict['NUM_ACTIONS']).astype(int)
		action = environment.rng.choice(inds, size = 1, p = probs)

		#print('action = ', action)

		new_obs, reward, done, info = environment.step(action)

		#print('new obs = ', new_obs)

		update_index = tuple(np.append(obs, action))

		t1_value_index = new_obs

		new_vals = q_table[tuple(new_obs)]	

		#print('new vals = ', new_vals)

		new_probs = np.exp(new_vals/TEMPERATURE)/(np.sum(np.exp(new_vals/TEMPERATURE)))
		new_exp_val = np.sum(new_probs*new_vals)

		q_table[update_index] = (1-alpha)*q_table[update_index] + alpha*(reward + config_dict['GAMMA']*new_exp_val)

		obs = new_obs

	all_Q[...,episode] = q_table


	np.save(str(seed)+"_all_Q.npy", all_Q)
	np.save(str(seed)+"_success_history.npy", environment.all_episode_success)

