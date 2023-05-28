from src.models.odor_senses import *
from src.models.gym_environment_class import FlyNavigator
#from stable_baselines3.deepq.policies import MlpPolicy
from stable_baselines3 import DQN
import stable_baselines3.common.utils
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np 
#from stable_baselines3 import DQN
import time
import sys 
import os

config_dict = {
	
	"NUM_ACTIONS": 4,
	"USE_COSINE_AND_SIN_THETA": True,
	"CONCENTRATION_BASE_THRESHOLD": 100, #this is the value that's good for movies. Do not change this to account for normalization-this happens internally.  
	"CONCENTRATION_THRESHOLD_STYLE": "fixed",
	"ODOR_FEATURES_CLASS": OdorFeatures,
	"DISCRETIZE_OBSERVABLES": False,
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
	"PER_STEP_REWARD": -1/60,
	"SOURCE_REWARD": 100,
	"WITH_ORIENTATION": False,
	"USE_MOVIE": True,
	"MOVIE_PATH": None,
	"MIN_FRAME": 500,
	"STOP_FRAME": 5000,
	"RESET_FRAME_RANGE": np.array([501,800]),
	"SOURCE_LOCATION_MM": np.array([30,90]),
	"GOAL_RADIUS_MM": 10, #success radius in mm
	"N_EPISODES" : 10000, # How many independently initialized runs to train on
	"MAX_ALPHA": 0.01, # Learning rate
	"MIN_ALPHA": 0.001,
	"GAMMA":0.95, # Reward temporal discount factor
	"MIN_EPSILON":0.01, # Asymptote of decaying exploration rate
	"MIN_RESET_X_MM": 40, # Initialization condition-minimum agent x in mm
	"INITIAL_MAX_RESET_X_MM": 300,
	"MAX_RESET_X_MM": 300, # Initialization condition-maximum agent x in mm
	"MIN_RESET_Y_MM": 0,
	"MAX_RESET_Y_MM": 180,
	"SOURCE_LOC_MM": np.array([30,90]), #Source location in mm
	"INIT_THETA_MIN": 0,
	"INIT_THETA_MAX": 2*np.pi,
	"TURN_ANG_SPEED_RAD_PER_S": 100*np.pi/180,
	"MIN_TURN_DUR_S": 0.18,
	"EXCESS_TURN_DUR_S": 0.18,
	"SHIFT_EPISODES": 10000,
	"RESET_X_SHIFT_MM": 5,
	"RENDER_VIDEO": True


}

seed = int(sys.argv[1])
rng = np.random.default_rng(seed)
plume_movie_path = os.path.join('..','src', 'data', 'plume_movies', 'intermittent_smoke.avi')

config_dict['MOVIE_PATH'] = plume_movie_path

env = FlyNavigator(rng = rng, config = config_dict)
env = DummyVecEnv([lambda: env])

num_steps = config_dict['STOP_FRAME']*config_dict['N_EPISODES']

models_dir = 'models_second_round/'
logdir = 'logs_second_round'

learning_rate = stable_baselines3.common.utils.get_linear_fn(start = config_dict['MAX_ALPHA'], end = config_dict['MIN_ALPHA'], end_fraction = 2/3)

model = DQN.load('models/'+str(seed)+"_after_99990000", env = env)

model.learning_rate = learning_rate
model.tensorboard_log = logdir
model.exploration_initial_eps = 0.05
model.exploration_fraction = 0.25

#make these directories

save_steps = config_dict['STOP_FRAME'] #roughly after every episode

for i in range(0, config_dict['N_EPISODES']):

	model.learn(total_timesteps=save_steps, reset_num_timesteps=False, tb_log_name = str(seed)+"_DQN_model")
	np.save(models_dir+str(seed)+"_reward_history.npy", np.array(environment.all_episode_rewards))
	np.save(models_dir+str(seed)+"_success_history.npy", np.array(environment.all_episode_success))
	model.save(models_dir+str(seed)+'after_'+str(config_dict['N_EPISODES']*i))
