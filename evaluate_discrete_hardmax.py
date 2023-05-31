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
	"N_EPISODES" : 100, # How many independently initialized runs to train on
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

q_table = np.load(str(seed)+"_all_Q.npy")[...,-1]

all_success = np.zeros(config_dict['N_EPISODES'])

for episode in range(0,config_dict['N_EPISODES']):

	obs = environment.reset()

	#print(obs)
	#print(environment.odor_features.concentration)

	done = False
	inds = np.arange(0,config_dict['NUM_ACTIONS']).astype(int)

	count = 0

	while not done:

		vals = q_table[tuple(obs)]

		possible_actions = np.argwhere(vals==np.amax(vals))
		action = environment.rng.choice(possible_actions)

		#print('action = ', action)

		new_obs, reward, done, info = environment.step(action)

		obs = new_obs


	if environment.reached_source:

		all_success[episode] = 1

np.savetxt(str(seed)+"_eval_all_success.txt", all_success)

