import sys
#sys.path.append('../../')
print(sys.path)

## Trains the baseline RNN on the odor plume using PPO on actor-critic MLP heads stemming from the RNN feature extractor
#from src.models.rnn_baseline import *
from src.environment.gym_environment_class import *
import os
import numpy as np
import gym
#from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv
import stable_baselines3.common.utils

plume_movie_path = os.path.join('..', 'src', 'data', 'plume_movies', 'intermittent_smoke.avi')

plume_dict = {
    "MM_PER_PX": 0.2,
    "MAX_CONCENTRATION": 255,
    "MOVIE_PATH": plume_movie_path,
	"MIN_FRAME": 500,
	"STOP_FRAME": 5000,
	"RESET_FRAME_RANGE": np.array([501, 800]),
	"SOURCE_LOCATION_MM": np.array([30,90]),
    "MIN_RESET_X_MM": 40, # Initialization condition-minimum agent x in mm
	"INITIAL_MAX_RESET_X_MM": 300,
	"MAX_RESET_X_MM": 300, # Initialization condition-maximum agent x in mm
	"MIN_RESET_Y_MM": 0,
	"MAX_RESET_Y_MM": 180,
    "SHIFT_EPISODES": 100,
	"RESET_X_SHIFT_MM": 5,
    "INIT_THETA_MIN": 0,
	"INIT_THETA_MAX": 2*np.pi,
	"PX_THRESHOLD": 100,
}

state_dict = {
    "USE_COSINE_AND_SIN_THETA": True,
    "DISCRETE_OBSERVABLES": False,
    "FEATURES": ['conc', 'grad', 'hrc'], ## see OdorFeatures class for options,
    "NORMALIZE_ODOR_FEATURES": True,
    "USE_BASE_THRESHOLD_FOR_MEAN": False,
	"CONCENTRATION_BASE_THRESHOLD": 100, #this is the value that's good for movies. Do not change this to account for normalization-this happens internally.  
	"CONCENTRATION_THRESHOLD_STYLE": "fixed",
	"THETA_DISCRETIZATION": 8, ## number of bins of discretizing theta
    "TIMESCALES_S": {"FILTER": 0.2, "THRESHOLD": 0.2}, ## timescales for temporal filtering and thresholding of concentration (in adaptive case)
    "FIX_ANTENNA": False, ## whether to fix the antenna to pointing upwind
}

output_dict = {
    "RENDER_VIDEO": 'dqn_agent_run.mp4', ## name of video file to render to
    'RECORD_SUCCESS': True, ## whether to record rewards and number of successful episodes
    "SAVE_DIRECTORY": './'
}

agent_dict = {
    "ANTENNA_LENGTH_MM": 0.41,
	"ANTENNA_WIDTH_MM": 0.21,
    "WALK_SPEED_MM_PER_S": 10,
	"DELTA_T_S": 1/60,
    "TURN_ANG_SPEED_RAD_PER_S": 100*np.pi/180,
	"MIN_TURN_DUR_S": 0.18,
	"EXCESS_TURN_DUR_S": 0.18,
    "GOAL_RADIUS_MM": 10, #success radius in mm
}

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

training_dict = {"GAMMA":0.99}


config_dict = {"agent": agent_dict, "plume": plume_dict, "state": state_dict, "output": output_dict, "reward": reward_dict, "training":training_dict}

N_EPISODES = 100

seed = int(sys.argv[1])
rng = np.random.default_rng(seed)
env = FlyNavigator(rng = rng, config = config_dict)
model_path = os.path.join('..','trained_models', 'dqn_no_temp_first_round_redo_060223', '7after_8000000.zip')

model = DQN.load(model_path, env = env)
success_arr = np.zeros(N_EPISODES)

num_cols = 7

data_arr = np.zeros((4500,num_cols,N_EPISODES))

for episode in range(0, N_EPISODES):

	obs = env.reset()
	done = False

	count = 0

	while not done:

		action = model.predict(obs)[0]

		new_row = np.zeros(num_cols)
		new_row[0:3] = obs[0:3]
		new_row[3] = env.fly_spatial_parameters.theta
		new_row[4] = action
		new_row[5:] = env.fly_spatial_parameters.position

		data_arr[count,:,episode] = new_row

		obs, reward, done, info = env.step(action)

		count+=1

	if env.reached_source:

		success_arr[episode] = 1

	if count<4500:

		data_arr[count:,:,episode] = np.nan

np.save(str(seed)+"_data_arr.npy", data_arr)
np.save(str(seed)+"_success_arr.npy", success_arr)




