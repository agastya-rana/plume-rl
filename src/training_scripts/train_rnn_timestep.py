import sys
print(sys.path)

## Trains the baseline RNN on the odor plume using PPO on actor-critic MLP heads stemming from the RNN feature extractor
from src.models.rnn_baseline import *
from src.environment.gym_environment_class import *
import os
import numpy as np
import gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecEnv
plume_movie_path = os.path.join('..', 'src', 'data', 'plume_movies', 'intermittent_smoke.avi')
plume_dict = {
    "MM_PER_PX": 0.2,
    "MAX_CONCENTRATION": 255,
    "PX_THRESHOLD": 100,
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
    "FEATURES": ['conc', 'grad', 'hrc'], ## see OdorFeatures class for options,
    "NORMALIZE_ODOR_FEATURES": True,
    "FIX_ANTENNA": False, ## whether to fix the antenna to pointing upwind
    "USE_BASE_THRESHOLD_FOR_MEAN": 100,  ## ReLUs the mean odor
    "DISCRETE_OBSERVABLES": False, ## Following features are only used in discrete cases
    "THETA_DISCRETIZATION": 8, ## number of bins of discretizing theta
	"CONCENTRATION_BASE_THRESHOLD": 100, ## Only applies for discrete features
	"CONCENTRATION_THRESHOLD_STYLE": "fixed", ## Only applies for discrete features
    "TIMESCALES_S": {"FILTER": 0.2, "THRESHOLD": 0.2}, ## timescales for temporal filtering and thresholding of concentration (in adaptive case)
}

output_dict = {
    'RECORD_SUCCESS': False, ## whether to record rewards and number of successful episodes
    'SAVE_DIRECTORY': os.path.join('..', 'trained_models', 'rnn'), ## directory to save model
    'RECORD_STATE_ACTION': 500, ## number of episodes to record state and action at each step in testing
}

agent_dict = {
    "ANTENNA_LENGTH_MM": 1,
	"ANTENNA_WIDTH_MM": 0.5,
    "WALK_SPEED_MM_PER_S": 10,
	"DELTA_T_S": 1/60,
    "TURN_ANG_SPEED_RAD_PER_S": 100*np.pi/180,
	"MIN_TURN_DUR_S": 0.18,
	"EXCESS_TURN_DUR_S": 0.18,
    "GOAL_RADIUS_MM": 10, #success radius in mm
    "INT_TIMESTEP": True, ## whether to use larger timesteps
    "INTEGRATED_DT": 1/10, ## timestep for integration
    "FEATURES_FILTER_SIZE": 5 ## size of filter for temporal filtering of features in units of original dt
}

reward_dict = {
	"SOURCE_REWARD": 10000,
	"PER_STEP_REWARD": -1/60,
	"WALL_PENALTY": -5000,
	"WALL_MAX_X_MM": 330,
	"WALL_MIN_X_MM": -10,
	"WALL_MIN_Y_MM": 0,
	"WALL_MAX_Y_MM": 180,
	"RADIAL_REWARD_SCALE": 0.1,
    "CONC_UPWIND_REWARD": 1/60,
    'CONC_REWARD': 1/60,
}

training_dict = {
    "MODEL_CLASS": RecurrentPPO,
    "POLICY": "MlpLstmPolicy",
    "N_EPISODES": 5000,
    "MAX_EPISODE_LENGTH": 5000,
    "GAMMA": 0.995, ## discount factor
    "GAE_LAMBDA": 0.95, ## GAE parameter
    "CLIP_RANGE": 0.2, ## clip range for PPO
    "VF_COEF": 0.5, ## value function coefficient in loss factor
    "ENT_COEF": 0.01, ## entropy coefficient in loss factor
    "LSTM_HIDDEN_SIZE": 16, ## size of LSTM hidden state
    "ACTOR_CRITIC_LAYERS": [16, 64, 64], ## MLP layers for actor-critic heads; first dimension should be lstm_hidden_size
    "N_ENVS": 8, ## number of parallel environments/CPU cores
    "N_STEPS": 512, ## number of steps per environment per update
    "MODEL_NAME": "rnn_shaping", ## name of model to save
    "TB_LOG": "./logs/rnn_shaping/", ## directory to save tensorboard logs
    "TEST_EPISODES": 1000, ## number of episodes to test the model
}

config_dict = {"agent": agent_dict, "plume": plume_dict, "state": state_dict, "output": output_dict, "training": training_dict, "reward": reward_dict}

if __name__ == "__main__":
    ## Train the model
    if len(sys.argv) > 1:
        config_dict["training"]["MODEL_NAME"] += '_' + sys.argv[1]
    model = train_model(config_dict)
    ## Test the model
    test_model(config_dict)