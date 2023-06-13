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
import optuna
from optuna.pruners import MedianPruner
from optuna.storages import JournalStorage, JournalFileStorage
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import pickle
import logging

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
    "DISCRETE_OBSERVABLES": False,
    "FEATURES": ['conc', 'grad', 'hrc'], ## see OdorFeatures class for options,
    "NORMALIZE_ODOR_FEATURES": True,
    "USE_BASE_THRESHOLD_FOR_MEAN": 100,  ## ReLUs the mean odor
	"CONCENTRATION_BASE_THRESHOLD": 100, #this is the value that's good for movies. Do not change this to account for normalization-this happens internally.  
	"CONCENTRATION_THRESHOLD_STYLE": "fixed",
	"THETA_DISCRETIZATION": 8, ## number of bins of discretizing theta
    "TIMESCALES_S": {"FILTER": 0.2, "THRESHOLD": 0.2}, ## timescales for temporal filtering and thresholding of concentration (in adaptive case)
    "FIX_ANTENNA": False, ## whether to fix the antenna to pointing upwind
}

output_dict = {
    "RENDER_VIDEO": 'rnn.mp4', ## name of video file to render to
    'RECORD_SUCCESS': False, ## whether to record rewards and number of successful episodes
    'SAVE_DIRECTORY': os.path.join('..', 'trained_models', 'rnn') ## directory to save model
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
    "INT_TIMESTEP": False, ## whether to use a longer timestep for integration
}

training_dict = {
    "MODEL_CLASS": RecurrentPPO,
    "POLICY": "MlpLstmPolicy",
    "N_EPISODES": 2000,
    "MAX_EPISODE_LENGTH": 5000,
    # "lr_schedule": "constant",
    # "learning_rate": 0.0001,
    "GAMMA": 0.995, ## change this later to match all upper case
    "GAE_LAMBDA": 0.95, ## GAE parameter
    "CLIP_RANGE": 0.2, ## clip range for PPO
    "VF_COEF": 0.5, ## value function coefficient in loss factor
    "ENT_COEF": 0.01, ## entropy coefficient in loss factor
    "LSTM_HIDDEN_SIZE": 32, ## size of LSTM hidden state
    "ACTOR_CRITIC_LAYERS": [32, 64, 64], ## MLP layers for actor-critic heads; first dimension should be lstm_hidden_size
    "N_ENVS": 8, ## number of parallel environments/CPU cores
    "N_STEPS": 512, ## number of steps per environment per update
    "MODEL_NAME": "rnn_shaping", ## name of model to save
    "TB_LOG": "./logs/rnn_shaping/", ## directory to save tensorboard logs
    "TEST_EPISODES": 2, ## number of episodes to test the model
}

## This code will do grid search over the following parameters
## 1. Reward shaping parameters
## a. CONC_UPWIND_REWARD
## b. CONC_REWARD
## c. RADIAL_REWARD_SCALE
## d. WALL_PENALTY
## e. SOURCE_REWARD
## 2. Training parameters
## a. gamma
## b. lstm_hidden_size
## c. actor_critic_layers
## Not using the below values, but ranges instead
conc_upwind_rewards = [0, 0.001, 0.01, 0.1] ## timestep reward is this times normalized concentration
conc_rewards = [0, 0.001, 0.01, 0.1] ## timestep reward is this times normalized concentration
radial_rewards = [0, 0.005, 0.05, 0.5] # 5 gives unit reward per timestep
wall_penalties = [-500, -2000, -10000]
source_rewards = [1000, 10000, 10000]

def objective(trial):
    ## Define the environment here
    rng = np.random.default_rng(seed=0)
    ## Define the model to be run
    ## Add tuned variables here
    conc_upwind_reward = trial.suggest_loguniform('conc_upwind_reward', 1e-4, 1e-1)
    conc_reward = trial.suggest_loguniform('conc_reward', 1e-4, 1e-1)
    radial_reward = trial.suggest_loguniform('radial_reward', 5e-4, 5)
    wall_penalty = trial.suggest_loguniform('wall_penalty', 5e2, 5e4)
    source_reward = trial.suggest_loguniform('source_reward', 1e3, 1e5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

    reward_dict = {
	"SOURCE_REWARD": source_reward,
	"PER_STEP_REWARD": -1/60,
	"IMPOSE_WALLS": True,
	"WALL_PENALTY": -wall_penalty,
	"WALL_MAX_X_MM": 330,
	"WALL_MIN_X_MM": -10,
	"WALL_MIN_Y_MM": 0,
	"WALL_MAX_Y_MM": 180,
	"RADIAL_REWARD": radial_reward,
    "CONC_UPWIND_REWARD": conc_upwind_reward,
    'CONC_REWARD': conc_reward,
    'MOTION_REWARD': 0,
    }
    config_dict = {"agent": agent_dict, "plume": plume_dict, "state": state_dict, "output": output_dict, "training": training_dict, "reward": reward_dict}
    model_class = training_dict["MODEL_CLASS"]
    # Create vectorized environments
    env = SubprocVecEnv([make_env(i, config_dict) for i in range(training_dict["N_ENVS"])])
    env = VecMonitor(env)
    model = model_class(training_dict["POLICY"], env, verbose=1, n_steps=training_dict["N_STEPS"], batch_size=training_dict["N_STEPS"]*training_dict["N_ENVS"], 
    policy_kwargs={"lstm_hidden_size": training_dict['LSTM_HIDDEN_SIZE'], "net_arch": training_dict['ACTOR_CRITIC_LAYERS']},
    gamma=training_dict['GAMMA'], gae_lambda=training_dict['GAE_LAMBDA'], clip_range=training_dict['CLIP_RANGE'], vf_coef=training_dict['VF_COEF'], ent_coef=training_dict['ENT_COEF'])
    model.learn(total_timesteps=training_dict['N_EPISODES']*training_dict['MAX_EPISODE_LENGTH'])
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=training_dict["TEST_EPISODES"], warn=False)
    ## Free up memory
    for environment in env.envs:
        environment.odor_plume = None
        environment.close()
        del environment
    del env
    return mean_reward

if __name__ == "__main__":
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f"rnn-reward-shaping_{sys.argv[1]}"  # Unique identifier of the study.
    storage = JournalStorage(JournalFileStorage(f"rnn-reward-journal_{sys.argv[1]}.log"))
    # Set n_jobs to -1 to use all cores
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=500, n_jobs=1)
    best_params = study.best_params
    best_value = study.best_value
    print(f"Best value: {best_value} \n Best params: {best_params}")

"""
if __name__ == "__main__":
    ## Train the model
    if len(sys.argv) > 1:
        config_dict["training"]["model_name"] += '_' + sys.argv[1]
        config_dict["output"]["RENDER_VIDEO"] = config_dict["output"]["RENDER_VIDEO"].split('.')[0] + '_' + sys.argv[1] + '.mp4'
        ## Set the reward parameters according to system argument index by modulo
        config_dict["reward"]["CONC_UPWIND_REWARD"] = conc_upwind_rewards[int(sys.argv[1]) % len(conc_upwind_rewards)]
        config_dict["reward"]["CONC_REWARD"] = conc_rewards[int(sys.argv[1] // len(conc_upwind_rewards)) % len(conc_rewards)]
        config_dict["reward"]["RADIAL_REWARD"] = radial_rewards[int(sys.argv[1] // (len(conc_upwind_rewards) * len(conc_rewards))) % len(radial_rewards)]
        config_dict["reward"]["WALL_PENALTY"] = wall_penalties[int(sys.argv[1] // (len(conc_upwind_rewards) * len(conc_rewards) * len(radial_rewards))) % len(wall_penalties)]
        config_dict["reward"]["SOURCE_REWARD"] = source_rewards[int(sys.argv[1] // (len(conc_upwind_rewards) * len(conc_rewards) * len(radial_rewards) * len(wall_penalties))) % len(source_rewards)]
        print("Reward parameters: ", config_dict["reward"])
    model = train_model(config_dict)
    ## Test the model
    test_model(config_dict)
"""