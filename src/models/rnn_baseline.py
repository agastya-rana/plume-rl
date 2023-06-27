from sb3_contrib import RecurrentPPO
from src.environment.gym_environment_class import *
from src.environment.env_variations import *
from src.environment.utilities import *
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import os
import numpy as np

# Helper function to create environments
def make_env(i, config_dict):
    def _init():
        if config_dict["agent"]["INT_TIMESTEP"]:
            return IntegratedTimestepsNavigator(np.random.default_rng(seed=i), config_dict)
        else:
            return FlyNavigator(np.random.default_rng(seed=i), config_dict)
    return _init


def train_model(config):
    print_dict(config)
    training_dict = config['training']
    store_config(config)
    ## Define the environment here
    ## Define the model to be run
    model_class = training_dict["MODEL_CLASS"]
    # Create vectorized environments
    env = SubprocVecEnv([make_env(i, config) for i in range(training_dict["N_ENVS"])])
    env = VecMonitor(env)#, info_keywords=('ep_rew_mean', 'ep_len_mean')); can add these if add corresponding params to info dict in step function of environment
    model = model_class(training_dict["POLICY"], env, verbose=1, n_steps=training_dict["N_STEPS"], batch_size=training_dict["N_STEPS"]*training_dict["N_ENVS"], 
    policy_kwargs={"lstm_hidden_size": training_dict['LSTM_HIDDEN_SIZE'], "net_arch": training_dict['ACTOR_CRITIC_LAYERS']},
    gamma=training_dict['GAMMA'], gae_lambda=training_dict['GAE_LAMBDA'], clip_range=training_dict['CLIP_RANGE'], vf_coef=training_dict['VF_COEF'], ent_coef=training_dict['ENT_COEF'],
    tensorboard_log=training_dict['TB_LOG'], learning_rate=training_dict['LEARNING_RATE'])
    # Train the model
    model.learn(total_timesteps=training_dict['N_EPISODES']*training_dict['MAX_EPISODE_LENGTH'], tb_log_name=training_dict['MODEL_NAME'])
    # Save the model
    model.save(os.path.join(config["output"]["SAVE_DIRECTORY"], training_dict['MODEL_NAME']))
    return model

def test_model(config):
    config["output"]["RECORD_SUCCESS"] = True
    model = RecurrentPPO.load(os.path.join(config["output"]["SAVE_DIRECTORY"], config["training"]['MODEL_NAME']))
    rng = np.random.default_rng(seed=1)
    render_env = FlyNavigator(rng, config)
    obs = render_env.reset()
    # cell and hidden state of the LSTM
    lstm_states = None
    # Episode start signal
    episode_start = True
    episode_no = 0
    num_record = config["output"]["RECORD_STATE_ACTION"]
    state_arr = np.empty((num_record, config["plume"]["STOP_FRAME"], render_env.obs_dim))
    action_arr = np.empty((num_record, config["plume"]["STOP_FRAME"], 4))
    while episode_no < config["training"]['TEST_EPISODES']:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
        if episode_no < num_record:
            ## Store state and action
            state_arr[episode_no, render_env.odor_plume.frame_number, :] = obs
            action_arr[episode_no, render_env.odor_plume.frame_number, :] = action
        obs, _, done, _ = render_env.step(action)
        if episode_no < 10:
            render_env.render()
        # If the episode is done, reset the environment (for vector environments, we don't need to reset manually)
        if done:
            obs = render_env.reset()
            episode_start = True
            episode_no += 1
        else:
            episode_start = False
    np.save(os.path.join(config["output"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_reward_history.npy"), np.array(render_env.all_episode_rewards))
    np.save(os.path.join(config["output"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_success_history.npy"), np.array(render_env.all_episode_success))
    ## Save state and action arrays
    np.save(os.path.join(config["output"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_state_history.npy"), state_arr)
    np.save(os.path.join(config["output"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_action_history.npy"), action_arr)
    print("Average reward: ", np.mean(render_env.all_episode_rewards))
    print("Average success: ", np.mean(render_env.all_episode_success))
    render_env.close()