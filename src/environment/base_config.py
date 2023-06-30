## File for base RL training configuration parameters:
import numpy as np
import os
## Sample plume dict for single video agent - default Fly Navigator
plume_dict = {
    "MM_PER_PX": 0.154,   
    "START_FRAME": 0, ## optional; default 0
    "STOP_FRAME": 3233,
	"RESET_FRAME_RANGE": np.array([0, 10]),
	"SOURCE_LOCATION_MM": np.array([14, 93.6]),
    "MOVIE_PATH": plume_movie_path,

    "RESET_BOX_MM": ((40, 200), (0, 180)), ## box of initialization positions ((x_min, x_max), (y_min, y_max)) in mm (later filtered to only those with odor signal)
    "INITIAL_MAX_RESET_X_MM": 200, ## optional; default is x_max of RESET_BOX_MM
    "RESET_X_SHIFT_MM": 5, ## optional; default 0
    "SHIFT_EPISODES": 100, ## optional; default 0
    "THETA_BOUNDS": (0, 2*np.pi), ## optional; default (0, 2*np.pi)

    "WALL_BOX_MM": ((-10, 330), (0, 180)), ## only required if wall_penalty is non-zero; box of wall positions ((x_min, x_max), (y_min, y_max)) in mm; 

    "MAX_CONCENTRATION": 255, ## optional; default 255
    "PX_THRESHOLD": 0, ## optional; default 0
    
    "PLUME_TYPE": "movie", ## optional; default "movie"
}

state_dict = {
    "FEATURES": ['conc', 'grad', 'hrc'], ## see OdorFeatures class for options,
    "HIST_LEN": 10, ## only required for environments with history

    "DISCRETE_OBSERVABLES": False, ## optional; default False
    "USE_COSINE_AND_SIN_THETA": True, ## optional; default True
    "THETA_DISCRETIZATION": 8, ## only required for discrete features option

    "DETECTION_THRESHOLD": 100,  ## optional; default 0; used to threshold left/right odor inputs and for discrete features
	"DETECTION_THRESHOLD_TYPE": "fixed", ## optional; default "fixed"; other option is "adaptive"
    "DETECTION_THRESHOLD_TIMESCALE_S": 0.2, ## only required for "adaptive"; timescale for adaptive thresholding (in seconds)

    "TAU_S": 0.2, ## only required for temporal features (currently only intermittency); timescale for temporal smoothing (in seconds)
    "MAX_T_L_S": 0.2, ## only required for temporal features (currently only T_L); maximum time since last whiff (in seconds)
    "FIX_ANTENNAE": False, ## optional; default False; whether to fix the antennae upwind
}

agent_dict = {
    "ANTENNA_LENGTH_MM": 1,
	"ANTENNA_WIDTH_MM": 0.5,
    "WALK_SPEED_MM_PER_S": 10,
	"DELTA_T_S": 1/60,
    "TURN_ANG_SPEED_RAD_PER_S": 100*np.pi/180,
	"MIN_TURN_DUR_S": 0.18,
	"EXCESS_TURN_DUR_S": 0.18,
    "GOAL_RADIUS_MM": 5, #success radius in mm

    ## Below are only required for environments with integrated timestep
    "INT_TIMESTEP": False,
    "INTEGRATED_DT_S": 0.2, ## agent timestep (in seconds)
    "FEATURES_FILTER_SIZE_S": 5, ## Exponential moving average timescale for integrating features (in seconds)
}

reward_dict = {
    ## All parameters below are optional; default values are 0; check _get_additional_rewards to see usage
	"SOURCE_REWARD": 400,
	"PER_STEP_REWARD": -1/60,
	"WALL_PENALTY": -400,
    "CONC_UPWIND_REWARD": 0,
    "UPWIND_REWARD": 0,
    "CONC_REWARD": 0,
	"RADIAL_REWARD": 10,
    "STRAY_REWARD": 0,
}

## Training dict is a function of the model used to train. Varies for DQN, RecurrentPPO, PPO.
## However, some parameters (those at the top) are required.
training_dict = {

    ## Save/logging parameters (directories, filenames, save/log frequencies + variables, callbacks, etc.)
    "MODEL_NAME": "dqn_newplume_radial", ## name of model to save; REQUIRED
    "TB_LOG": "../logs/dqn_hist/", ## directory to save tensorboard logs
    'SAVE_DIRECTORY': os.path.join('.', 'trained_models'), ## directory to save model; REQUIRED
    'RECORD_SUCCESS': False, ## optional; default False
    'RECORD_STATE_ACTION': 1000, ## number of episodes to record state and action at each step in testing; optional
    "RENDER_VIDEO": True, ## whether to render the video of the agent; optional; default False

    ## Training/testing timesteps/epochs
    "N_EPISODES": 10000,
    "MAX_EPISODE_LENGTH": 5000,
    "TEST_EPISODES": 1000, ## number of episodes to test the model


    ## Hyperparameters (learning rates, gamma, epsilon, etc.)
    "MAX_ALPHA": 0.001,
    "MIN_ALPHA": 0.0001,
    "GAMMA": 0.9999,
    "MIN_EPSILON": 0.05,
    "LEARNING_END_FRACTION": 1/5,

    ## Network architecture; required
    "N_HIDDEN_UNITS": 32, ## number of hidden units in MLP layers
    "N_HIDDEN_LAYERS": 3, ## number of hidden layers in MLP
    "FEATURES_EXTRACTOR_CLASS": None, ## class of features extractor; optional; default None

}

config_dict = {"agent": agent_dict, "plume": plume_dict, "state": state_dict, "training": training_dict, "reward": reward_dict}