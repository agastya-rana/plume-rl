from src.models.odor_senses import *
from src.models.gym_environment_class import FlyNavigator
#from stable_baselines3.deepq.policies import MlpPolicy
from stable_baselines3 import DQN
import stable_baselines3.common.utils

import numpy as np 
#from stable_baselines3 import DQN
import time
import sys 
import os

config_dict = {
	
	"NUM_ACTIONS": 4,
	"OBSERVATION_DIMENSION": 7,
	"CONCENTRATION_BASE_THRESHOLD": 0.5, #100 good for videos, around 1 good for plume sims-remember to change!
	"CONCENTRATION_THRESHOLD_STYLE": "fixed",
	"ODOR_FEATURES_CLASS": OdorFeatures, #note, not an instantiation
	"TEMPORAL_FILTER_TIMESCALE_S": 1,
	"TEMPORAL_THRESHOLD_ADAPTIVE_TIMESCALE_S":5,
	"TEMPORAL_FILTER_ALL":False,
	"MM_PER_PX": 0.2,
	"ANTENNA_LENGTH_MM": 1,
	"ANTENNA_WIDTH_MM": 0.5,
	"MAX_CONCENTRATION": 255,
	"NORMALIZE_ODOR_FEATURES": True,
	"WALK_SPEED_MM_PER_S": 10,
	"DELTA_T_S": 1/60,
	"PER_STEP_REWARD": 0,
	"WITH_ORIENTATION": False,
	"USE_MOVIE": True,
	"MOVIE_PATH": None,
	"MIN_FRAME": 500,
	"STOP_FRAME": 5000,
	"RESET_FRAME_RANGE": np.array([501,801]),
	"SOURCE_LOCATION_MM": np.array([30,90]),
	"GOAL_RADIUS_MM": 10, #success radius in mm
	"N_EPISODES" : 2000, # How many independently initialized runs to train on
	"MAX_ALPHA": 0.1, # Learning rate
	"MIN_ALPHA": 0.0001,
	"GAMMA":0.95, # Reward temporal discount factor
	"MIN_EPSILON":0.01, # Asymptote of decaying exploration rate
	"MIN_RESET_X_MM": 40, # Initialization condition-minimum agent x in mm
	"INITIAL_MAX_RESET_X_MM": 45,
	"MAX_RESET_X_MM": 300, # Initialization condition-maximum agent x in mm
	"MIN_RESET_Y_MM": 0,
	"MAX_RESET_Y_MM": 180,
	"WALL_MIN_X_MM": 0,
	"WALL_MAX_X_MM": 300,
	"WALL_MIN_Y_MM": 0,
	"WALL_MAX_Y_MM": 175,
	"SOURCE_LOC_MM": np.array([30,90]), #Source location in mm
	"INIT_THETA_MIN": 0,
	"INIT_THETA_MAX": 2*np.pi,
	"TURN_ANG_SPEED_RAD_PER_S": 100*np.pi/180,
	"MIN_TURN_DUR_S": 0.18,
	"EXCESS_TURN_DUR_S": 0.18,
	"SHIFT_EPISODES": 100,
	"RESET_X_SHIFT_MM": 5

}

seed = int(sys.argv[1])
rng = np.random.default_rng(seed)
plume_movie_path = os.path.join('..','src', 'data', 'plume_movies', 'intermittent_smoke.avi')

config_dict['MOVIE_PATH'] = plume_movie_path
config_dict['N_EPISODES'] = 10000

environment = FlyNavigator(rng = rng, config = config_dict)

num_steps = config_dict['STOP_FRAME']*config_dict['N_EPISODES']

models_dir = 'models'
logdir = 'logs'

learning_rate = stable_baselines3.common.utils.get_linear_fn(start = config_dict['MAX_ALPHA'], end = config_dict['MIN_ALPHA'], end_fraction = 2/3)

model = DQN("MlpPolicy", environment, verbose = 1, tensorboard_log=logdir, gamma = config_dict['GAMMA'], 
	exploration_final_eps = config_dict['MIN_EPSILON'], seed = seed, learning_rate=learning_rate)

#make these directories

save_steps = config_dict['STOP_FRAME'] #roughly after every episode

for i in range(0, config_dict['N_EPISODES']):

	model.learn(total_timesteps=save_steps, reset_num_timesteps=False, tb_log_name = str(seed)+"_DQN_model")
	np.save('models/'+str(seed)+"_reward_history.npy", np.array(environment.all_episode_rewards))
	np.save('models/'+str(seed)+"_success_history.npy", np.array(environment.all_episode_success))
	model.save('models/'+'after_'+str(config_dict['N_EPISODES']*i))







