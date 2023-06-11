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
from stable_baselines3.common.callbacks import BaseCallback

plume_movie_path = os.path.join('..', 'src', 'data', 'plume_movies', 'intermittent_smoke.avi')

plume_dict = {
	"MM_PER_PX": 0.2,
	"MAX_CONCENTRATION": 255,
	"PLUME_TYPE": 'movie',
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
	"CONCENTRATION_BASE_THRESHOLD": 100, #this is the value that's good for movies. Do not change this to account for normalization-this happens internally.
	"USE_BASE_THRESHOLD_FOR_MEAN": False,  
	"CONCENTRATION_THRESHOLD_STYLE": "fixed",
	"THETA_DISCRETIZATION": 8, ## number of bins of discretizing theta
	"TIMESCALES_S": {"FILTER": 0.2, "THRESHOLD": 0.2}, ## timescales for temporal filtering and thresholding of concentration (in adaptive case)
	"FIX_ANTENNA": False, ## whether to fix the antenna to pointing upwind
}

output_dict = {
	"RENDER_VIDEO": 'dqn_agent_run.mp4', ## name of video file to render to
	"RECORD_SUCCESS": True, ## whether to record rewards and number of successful episodes
	"SAVE_DIRECTORY": './',
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
	"WALL_PENALTY": 0,
	"WALL_MAX_X_MM": 330,
	"WALL_MIN_X_MM": -10,
	"WALL_MIN_Y_MM": 0,
	"WALL_MAX_Y_MM": 180,
	"USE_RADIAL_REWARD": False,
	"RADIAL_REWARD": 0,
	"POTENTIAL_SHAPING": True,
	"UPWIND_REWARD": 1,
	"CONC_REWARD": 0,
	"CONC_UPWIND_REWARD":0,
	"MOTION_REWARD": 0
}

training_dict = {

"MAX_ALPHA": 0.01,
"MIN_ALPHA": 0.0001,
"GAMMA": 0.99,
"MIN_EPSILON":0.01,
"LEARNING_END_FRACTION": 2/3,
"LOGDIR": 'logs',
"N_EPISODES": 10000,
"model_name": 'DQN_no_temp'


}

config_dict = {"agent": agent_dict, "plume": plume_dict, "state": state_dict, "output": output_dict, "training": training_dict, "reward": reward_dict}


seed = int(sys.argv[1])
rng = np.random.default_rng(seed)

environment = FlyNavigator(rng = rng, config = config_dict)

models_dir = 'models/'

learning_rate = stable_baselines3.common.utils.get_linear_fn(start = training_dict['MAX_ALPHA'], 
 end = training_dict['MIN_ALPHA'], end_fraction = training_dict['LEARNING_END_FRACTION'])

model = DQN("MlpPolicy", environment, verbose = 1, tensorboard_log=None, gamma = training_dict['GAMMA'], 
	exploration_final_eps = training_dict['MIN_EPSILON'], seed = seed, learning_rate=learning_rate)


class CustomCallback(BaseCallback):

	def __init__(self, save_freq, save_dir, seed = seed, verbose=True):

		super(CustomCallback, self).__init__(verbose)
		self.save_freq = save_freq
		self.save_dir = save_dir
		self.seed = seed

	def _on_step(self) -> bool:
		if self.n_calls % self.save_freq == 0:
			n_eps = len(self.model.env.all_episode_rewards)
			save_string = str(self.seed)+"_after_"+str(n_eps)
			self.model.save(self.save_dir+save_string)
			np.save(self.save_dir+str(self.seed)+"_reward_history.npy", np.array(self.model.env.all_episode_rewards))
			np.save(self.save_dir+str(self.seed)+"_success_history.npy", np.array(self.model.env.all_episode_success))


callback = CustomCallback(save_freq = 500000, save_dir = 'models/')

model.learn(total_timesteps=25000000, reset_num_timesteps=False, callback=callback)











