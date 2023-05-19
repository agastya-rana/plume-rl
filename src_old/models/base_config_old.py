##File for base RL training configuration parameters:
##Dictionary should be imported only by run script and specific parameters changed there
import numpy as np 


config_dict = {
	
	"CONCENTRATION_THRESHOLD": 100, #100 good for videos, around 1 good for plume sims-remember to change!
	"MM_PER_PX": 0.2,
	"ANTENNA_LENGTH_MM": 1,
	"ANTENNA_WIDTH_MM": 0.5,
	"WALK_SPEED_MM_PER_S": 10,
	"DELTA_T_S": 1/60,
	"WITH_ORIENTATION": False,
	"USE_MOVIE": True,
	"MOVIE_PATH": None,
	"MIN_FRAME": 500,
	"STOP_FRAME": 5000,
	"RESET_FRAME_RANGE": np.array([501,801]),
	"PACKET_SIM_DICT":None,
	"RNG_SEED": 0,
	"SOURCE_LOCATION_MM": np.array([30,90]),
	"GOAL_RADIUS_MM": 10, #success radius in mm
	"N_EPISODES" : 2000, # How many independently initialized runs to train on
    "MAX_ALPHA": 0.2, # Learning rate
    "MIN_ALPHA": 0.005,
    "GAMMA":0.95, # Reward temporal discount factor
    "MAX_EPSILON":1, # Starting exploration rate
    "MIN_EPSILON":0.01, # Asymptote of decaying exploration rate
    "DECAY":0.05, # Rate of exploration decay
    "MIN_RESET_X_MM": 55, # Initialization condition-minimum agent x in mm
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
    "ODOR_MOTION_CALCULATION_STYLE": 'frame_correlation',
    "GRADIENT_CALCULATION_STYLE": 'regression',



}