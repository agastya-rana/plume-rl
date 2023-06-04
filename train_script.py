import sys
#sys.path.append('../../')
print(sys.path)

## Trains the baseline RNN on the odor plume using PPO on actor-critic MLP heads stemming from the RNN feature extractor
from src.models.rnn_baseline import *
from src.environment.gym_environment_class import *
import os
import numpy as np
import gym
#from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv
plume_movie_path = os.path.join('.', 'src', 'data', 'plume_movies', 'intermittent_smoke.avi')

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
}

state_dict = {
    "USE_COSINE_AND_SIN_THETA": True,
    "DISCRETE_OBSERVABLES": False,
    "FEATURES": ['conc', 'grad', 'hrc'], ## see OdorFeatures class for options,
    "NORMALIZE_ODOR_FEATURES": True,
	"CONCENTRATION_BASE_THRESHOLD": 100, #this is the value that's good for movies. Do not change this to account for normalization-this happens internally.  
	"CONCENTRATION_THRESHOLD_STYLE": "fixed",
	"THETA_DISCRETIZATION": 8, ## number of bins of discretizing theta
    "TIMESCALES_S": {"FILTER": 0.2, "THRESHOLD": 0.2}, ## timescales for temporal filtering and thresholding of concentration (in adaptive case)
    "FIX_ANTENNA": False, ## whether to fix the antenna to pointing upwind
}

output_dict = {
    "RENDER_VIDEO": None, ## name of video file to render to
    'RECORD_SUCCESS': True ## whether to record rewards and number of successful episodes
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

training_dict = {

"MAX_ALPHA": 0.1,
"MIN_ALPHA": 0.001,
"GAMMA": 0.99,
"MIN_EPSILON":0.01,
"LEARNING_END_FRACTION": 2/3,
"LOGDIR": 'logs',
"N_EPISODES": 10000


}

config_dict = {"agent": agent_dict, "plume": plume_dict, "state": state_dict, "output": output_dict, "training": training_dict, "reward": reward_dict}


seed = int(sys.argv[1])
rng = np.random.default_rng(seed)

environment = FlyNavigator(rng = rng, config = config_dict)

models_dir = 'models'
logdir = 'logs'

learning_rate = stable_baselines3.common.utils.get_linear_fn(start = training_dict['MAX_ALPHA'], 
 end = training_dict['MIN_ALPHA'], end_fraction = training_dict['LEARNING_END_FRACTION'])

model = DQN("MlpPolicy", environment, verbose = 1, tensorboard_log=training_dict['LOGDIR'], gamma = training_dict['GAMMA'], 
	exploration_final_eps = training_dict['MIN_EPSILON'], seed = seed, learning_rate=learning_rate)

#make these directories

save_steps = plume_dict['STOP_FRAME'] #roughly after every episode

for i in range(0, training_dict['N_EPISODES']):

	model.learn(total_timesteps=save_steps, reset_num_timesteps=False, tb_log_name = str(seed)+"_DQN_model")
	np.save('models/'+str(seed)+"_reward_history.npy", np.array(environment.all_episode_rewards))
	np.save('models/'+str(seed)+"_success_history.npy", np.array(environment.all_episode_success))

	if (i+1) % 100 == 0:

		model.save(models_dir+str(seed)+'after_'+str(i))







