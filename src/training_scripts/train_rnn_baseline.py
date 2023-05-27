import sys
#sys.path.append('../../')
print(sys.path)

## Trains the baseline RNN on the odor plume using PPO on actor-critic MLP heads stemming from the RNN feature extractor
from src.models.rnn_baseline import *
from src.environment.gym_environment_class import *
import os
import numpy as np
import gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecEnv
plume_movie_path = os.path.join('.', 'src', 'data', 'plume_movies', 'intermittent_smoke.avi')

plume_dict = {
    "MM_PER_PX": 0.2,
    "MAX_CONCENTRATION": 255,
    "MOVIE_PATH": plume_movie_path,
	"MIN_FRAME": 500,
	"STOP_FRAME": 4000,
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
    "DISCRETE_OBSERVABLES": False,
    "FEATURES": ['conc', 'grad', 'hrc'], ## see OdorFeatures class for options,
    "NORMALIZE_ODOR_FEATURES": True,
	"CONCENTRATION_BASE_THRESHOLD": 100, #this is the value that's good for movies. Do not change this to account for normalization-this happens internally.  
	"CONCENTRATION_THRESHOLD_STYLE": "fixed",
	"THETA_DISCRETIZATION": 8, ## number of bins of discretizing theta
    "TIMESCALES_S": {"FILTER": 0.2, "THRESHOLD": 0.2} ## timescales for temporal filtering and thresholding of concentration (in adaptive case)
}

output_dict = {
    "RENDER_VIDEO": 'rnn_cont.mp4', ## name of video file to render to
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
    "model_class": RecurrentPPO,
    "policy": "MlpLstmPolicy",
    "n_episodes": 20,
    "max_episode_length": 5000,
    # "lr_schedule": "constant",
    # "learning_rate": 0.0001,
    "gamma": 0.99, ## discount factor
    "gae_lambda": 0.95, ## GAE parameter
    "clip_range": 0.2, ## clip range for PPO
    "vf_coef": 0.5, ## value function coefficient in loss factor
    "ent_coef": 0.01, ## entropy coefficient in loss factor
    "lstm_hidden_size": 64, ## size of LSTM hidden state
    "actor_critic_layers": [64, 64], ## MLP layers for actor-critic heads; first dimension should be lstm_hidden_size
    "n_envs": 8, ## number of parallel environments/CPU cores
    "n_steps": 512, ## number of steps per environment per update
    "model_name": "ppo_recurrent_cont", ## name of model to save
    "tensorboard_log": "./logs/ppo_recurrent_cont/", ## directory to save tensorboard logs
}

config_dict = {"agent": agent_dict, "plume": plume_dict, "state": state_dict, "output": output_dict, "training": training_dict}

if __name__ == "__main__":
    ## Train the model
    model = train_model(config_dict)
    ## Test the model
    test_model(model, config_dict)