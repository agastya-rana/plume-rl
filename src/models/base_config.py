##File for base RL training configuration parameters:
##Dictionary should be imported only by run script and specific parameters changed there
import numpy as np 
from src.models.odor_senses import OdorFeatures


config_dict = {
	
	"NUM_ACTIONS": 4,
<<<<<<< HEAD
    "OBSERVABLES": ("conc", "grad", "hrc", "int", "t_L_prev", "t_L_current", "theta"),
    "NUM_ACTIONS": 4,
    "OBSERVATION_DIMENSION": 7,
    "CONCENTRATION_BASE_THRESHOLD": 0.5, #100 good for videos, around 1 good for plume sims-remember to change!
    "CONCENTRATION_THRESHOLD_STYLE": "fixed",
    "ODOR_FEATURES_CLASS": OdorFeatures,
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
    "USE_MOVIE": True,
    "MOVIE_PATH": None,
    "MIN_FRAME": 500,
    "STOP_FRAME": 5000,
    "RESET_FRAME_RANGE": np.array([501,801]),
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
    "WALL_MIN_X_MM": 0,
    "WALL_MAX_X_MM": 300,
    "WALL_MIN_Y_MM": 0,
    "WALL_MAX_Y_MM": 175,
    "SOURCE_LOC_MM": np.array([30,90]), #Source location in mm
    "INIT_THETA_MIN": 0,
    "INIT_THETA_MAX": 2*np.pi,
    "TURN_ANG_SPEED_RAD_PER_S": 100*np.pi/180,
    "MIN_TURN_DUR_S": 0.18,
    "EXCESS_TURN_DUR_S": 0.18,
    "SHIFT_EPISODES": 100,
    "RESET_X_SHIFT_MM": 5
=======
	"USE_COSINE_AND_SIN_THETA": True,
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
	"USE_MOVIE": True,
	"MOVIE_PATH": None,
	"MIN_FRAME": 500,
	"STOP_FRAME": 5000,
	"RESET_FRAME_RANGE": np.array([501,801]),
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


>>>>>>> master
}