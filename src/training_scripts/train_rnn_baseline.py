## Trains the baseline RNN on the odor plume using PPO on actor-critic MLP heads stemming from the RNN feature extractor
from src.models.rnn_baseline import *
from src.models.gym_environment_class import *
from src.models.base_config import *
import os
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
plume_movie_path = os.path.join('.', 'src', 'data', 'plume_movies', 'intermittent_smoke.avi')

plume_dict = {
    "MM_PER_PX": 0.2,
    "MAX_CONCENTRATION": 255,
    "MOVIE_PATH": plume_movie_path,
	"MIN_FRAME": 500,
	"STOP_FRAME": 1000,
	"RESET_FRAME_RANGE": np.array([500, 700]),
	"SOURCE_LOCATION_MM": np.array([30,90]),
    "MIN_RESET_X_MM": 40, # Initialization condition-minimum agent x in mm
	"INITIAL_MAX_RESET_X_MM": 150,
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
	"CONCENTRATION_BASE_THRESHOLD": 100, #this is the value that's good for movies. Do not change this to account for normalization-this happens internally.  
	"CONCENTRATION_THRESHOLD_STYLE": "fixed",
	"FEATURES": ['conc', 'grad', 'hrc'], ## see OdorFeatures class for options
	"THETA_DISCRETIZATION": 6, ## number of bins of discretizing theta
    "NORMALIZE_ODOR_FEATURES": True,
    "TIMESCALES_S": {"FILTER": 0.2, "THRESHOLD": 0.2}, ## timescales for temporal filtering and thresholding of concentration (in adaptive case)
    "DISCRETE_OBSERVABLES": False
}

output_dict = {
    "RENDER_VIDEO": 'trial.mp4', ## name of video file to render to
    'RECORD_SUCCESS': False ## whether to record rewards and number of successful episodes
}

agent_dict = {
    "ANTENNA_LENGTH_MM": 1,
	"ANTENNA_WIDTH_MM": 0.5,
    "WALK_SPEED_MM_PER_S": 10,
	"DELTA_T_S": 1/60,
    "TURN_ANG_SPEED_RAD_PER_S": 100*np.pi/180,
	"MIN_TURN_DUR_S": 0.18,
	"EXCESS_TURN_DUR_S": 0.18,
    "PER_STEP_REWARD": -1,
    "GOAL_REWARD": 1000,
    "GOAL_RADIUS_MM": 10, #success radius in mm
}

training_dict = {
    "N_EPISODES": 10000,
    "GAMMA": 0.95, # Reward temporal discount factor
	"MIN_EPSILON":0.01, # Asymptote of decaying exploration rate
    "MAX_ALPHA": 0.1, # Learning rate
	"MIN_ALPHA": 0.0001, # Learning rate
    'RNN_HIDDEN_SIZE': 128
}

config_dict = {"agent": agent_dict, "plume": plume_dict, "state": state_dict, "output": output_dict, "training": training_dict}




## Define the environment here
rng = np.random.default_rng(seed=0)
environment = FlyNavigator(rng, config_dict)
## Define the model to be run
model = RecurrentPPO("MlpLstmPolicy", environment, verbose=1, n_steps=128, batch_size=128*8, policy_kwargs={"lstm_hidden_size": training_dict['RNN_HIDDEN_SIZE']})
# Train the model
model.learn(total_timesteps=training_dict['N_EPISODES']*config_dict["plume"]["STOP_FRAME"])
# Save the model
model.save("ppo_recurrent")

# Create a list of environments
num_envs = 8
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