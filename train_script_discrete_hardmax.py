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


reward_dict = {
	
	"SOURCE_REWARD": 500,
	"PER_STEP_REWARD": -1/60,
	"IMPOSE_WALLS": True,
	"WALL_PENALTY": -20,
	"WALL_MAX_X_MM": 330,
	"WALL_MIN_X_MM": -10,
	"WALL_MIN_Y_MM": 0,
	"WALL_MAX_Y_MM": 180,
	"USE_RADIAL_REWARD": True,
	"RADIAL_REWARD_SCALE": 5,

}


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
	"WITH_ORIENTATION": False,
	"USE_MOVIE": True,
	"MOVIE_PATH": None,
	"MIN_FRAME": 500,
	"STOP_FRAME": 5000,
	"RESET_FRAME_RANGE": np.array([501,800]),
	"SOURCE_LOCATION_MM": np.array([30,90]),
	"GOAL_RADIUS_MM": 10, #success radius in mm
	"N_EPISODES" : 10000, # How many independently initialized runs to train on
	"MAX_ALPHA": 0.1, # Learning rate
	"MIN_ALPHA": 0.001,
	"SOFTMAX": True,
	"GAMMA":0.99, # Reward temporal discount factor
	"MIN_EPSILON":0.01, # Asymptote of decaying exploration rate
	"MIN_RESET_X_MM": 40, # Initialization condition-minimum agent x in mm
	"INITIAL_MAX_RESET_X_MM": 100,
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
	"reward_dict": reward_dict

}



seed = int(sys.argv[1])
rng = np.random.default_rng(seed)
plume_movie_path = os.path.join('..','src', 'data', 'plume_movies', 'intermittent_smoke.avi')
#plume_movie_path = '../src/data/plume_movies/intermittent_smoke.avi'

config_dict['MOVIE_PATH'] = plume_movie_path

environment = FlyNavigator(rng = rng, config = config_dict)

#TEMPERATURE = config_dict['SOURCE_REWARD']/200

q_shape = np.append(environment.observation_space.nvec,environment.action_space.n)
q_table = np.zeros(shape=q_shape)
#print(q_shape)


alpha_fraction = 3/4
alpha_arr = np.zeros(config_dict['N_EPISODES'])
alpha_change_num = int(config_dict['N_EPISODES']*alpha_fraction)

epsilon_fraction = 0.3
epsilon_arr = np.zeros(config_dict['N_EPISODES'])
epsilon_change_num = int(config_dict['N_EPISODES']*epsilon_fraction)

alpha_inds = np.arange(0, alpha_change_num)
epsilon_inds = np.arange(0, epsilon_change_num)

alpha_vals = config_dict['MAX_ALPHA'] - (config_dict['MAX_ALPHA']-config_dict['MIN_ALPHA'])/alpha_change_num * alpha_inds
epsilon_vals = 1 - (1-config_dict['MIN_EPSILON'])/epsilon_change_num * epsilon_inds


alpha_arr[0:alpha_change_num] = alpha_vals
alpha_arr[alpha_change_num:] = config_dict['MIN_ALPHA'] 

epsilon_arr[0:epsilon_change_num] = epsilon_vals
epsilon_arr[epsilon_change_num:] = config_dict['MIN_EPSILON']

#episode_incrementer = 0

all_Q_shape = np.append(q_shape, config_dict['N_EPISODES'])
all_Q = np.zeros(all_Q_shape)

for episode in range(0,config_dict['N_EPISODES']):

	obs = environment.reset()

	#print(obs)
	#print(environment.odor_features.concentration)

	done = False

	ranfs = environment.rng.uniform(size = config_dict['STOP_FRAME'])
	alpha = alpha_arr[episode]
	epsilon = epsilon_arr[episode]
	inds = np.arange(0,config_dict['NUM_ACTIONS']).astype(int)

	count = 0

	while not done:

		vals = q_table[tuple(obs)]

		rand_unif = ranfs[count]
		explore = epsilon < rand_unif

		if explore:

			action = environment.rng.choice(inds)

		else:

			possible_actions = np.argwhere(vals==np.amax(vals))
			action = environment.rng.choice(possible_actions)

		#print('action = ', action)

		new_obs, reward, done, info = environment.step(action)

		#print('new obs = ', new_obs)

		update_index = tuple(np.append(obs, action))

		t1_value_index = new_obs

		new_vals = q_table[tuple(new_obs)]	

		#print('new vals = ', new_vals)
		max_val = np.amax(new_vals)

		q_table[update_index] = (1-alpha)*q_table[update_index] + alpha*(reward + config_dict['GAMMA']*max_val)

		obs = new_obs

		count+=1

	all_Q[...,episode] = q_table


	np.save(str(seed)+"_all_Q.npy", all_Q)
	np.save(str(seed)+"_success_history.npy", environment.all_episode_success)

