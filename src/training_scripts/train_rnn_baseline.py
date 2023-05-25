## Trains the baseline RNN on the odor plume using PPO on actor-critic MLP heads stemming from the RNN feature extractor
from src.models.rnn_baseline import *
from src.models.gym_environment_class import *
from src.models.base_config import *
import os
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
plume_movie_path = os.path.join('.', 'src', 'data', 'plume_movies', 'intermittent_smoke.avi')


## Define the configuration dictionary
config_dict = {
	"NUM_ACTIONS": 4,
	"USE_COSINE_AND_SIN_THETA": False,
	"CONCENTRATION_BASE_THRESHOLD": 100, #this is the value that's good for movies. Do not change this to account for normalization-this happens internally.  
	"CONCENTRATION_THRESHOLD_STYLE": "fixed",
	"ODOR_FEATURES_CLASS": OdorFeatures,
	"DISCRETIZE_OBSERVABLES": False,
	"THETA_DISCRETIZATION": 6, #number of bins of discretizing theta
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
	"MOVIE_PATH": None,
	"MIN_FRAME": 500,
	"STOP_FRAME": 5000,
	"RESET_FRAME_RANGE": np.array([500, 700]),
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
	"SOURCE_LOC_MM": np.array([30,90]), #Source location in mm
	"INIT_THETA_MIN": 0,
	"INIT_THETA_MAX": 2*np.pi,
	"TURN_ANG_SPEED_RAD_PER_S": 100*np.pi/180,
	"MIN_TURN_DUR_S": 0.18,
	"EXCESS_TURN_DUR_S": 0.18,
	"SHIFT_EPISODES": 100,
	"RESET_X_SHIFT_MM": 5
	"RENDER_VIDEO": True,
}

config_dict['N_EPISODES'] = 1000000
config_dict['MOVIE_PATH'] = plume_movie_path
config_dict['MIN_FRAME'] = 500
config_dict['STOP_FRAME'] = 550
config_dict['RESET_FRAME_RANGE'] = np.array([501,509])
config_dict['RNN_HIDDEN_SIZE'] = 128
config_dict['VIDEO'] = True

## Define the environment here
rng = np.random.default_rng(seed=0)
environment = FlyNavigator(rng, config_dict)
## Define the model to be run
model = RecurrentPPO("MlpLstmPolicy", environment, verbose=1, n_steps=128, batch_size=128*8, policy_kwargs={"lstm_hidden_size": config_dict['RNN_HIDDEN_SIZE']})
# Train the model
model.learn(total_timesteps=1000)
# Save the model
model.save("ppo_recurrent")

# Create a list of environments
num_envs = 1
vec_env = model.get_env()
obs = vec_env.reset()
# cell and hidden state of the LSTM
lstm_states = None
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
for _ in range(500):
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    episode_starts = dones
    vec_env.render("human")
print('here')
vec_env.close()