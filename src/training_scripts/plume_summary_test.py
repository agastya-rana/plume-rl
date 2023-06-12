from src.models.plume_summary import PlumeSummary
import os
import numpy as np
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
training_dict = {}
config_dict = {"agent": agent_dict, "plume": plume_dict, "state": state_dict, "output": output_dict, "training": training_dict, "reward": reward_dict}

rbins = [20*i for i in range(8)]
thetabins = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
summary = PlumeSummary(config_dict, rbins, thetabins, n_points=500, samples_per_point=500)
summary.plot()