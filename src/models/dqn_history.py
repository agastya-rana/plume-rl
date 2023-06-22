## This file contains the code for training a DQN agent with history dependence using the HistoryNavigator environment

from stable_baselines3 import DQN
from src.environment.env_variations import HistoryNavigator
from src.environment.utilities import *
import stable_baselines3
import numpy as np
import os
import sys
import gym.spaces as spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch.nn.functional as F
import torch

class FilterExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_heads=3, n_odor=1, history_len=10, theta_dim=2):
        self.n_odor = n_odor
        self.n_heads = n_heads
        features_dim = n_heads*n_odor + theta_dim ## The features dimension is the number of neurons in the first layer + the number of theta values
        self.theta_dim = theta_dim
        self.history_len = history_len + 1 ## + 1 because the current timestep is also included for easier manipulation in the forward method
        super().__init__(observation_space, features_dim)
        # Define the first layer with n_heads*n_odor neurons that is a linear filter with offset
        self.filter = nn.ModuleList([nn.Linear(self.history_len, 1) for _ in range(n_heads*n_odor)])

    def forward(self, observations):
        ## Remove the theta values from the observations
        x = observations[:, :-self.theta_dim]
        x = x.view(-1, self.history_len, self.n_odor) ## Reshape to (batch_size, history_len, n_odor)
        x = [F.relu(fc(x[:, :, i // self.n_heads])) for i, fc in enumerate(self.filter)] ## List of tensors of shape (batch_size, 1)
        x = torch.cat(x, dim=1) ## Concatenate along the last dimension to get a tensor of shape (batch_size, n_heads*n_odor)
        ## Add back the theta values to x and return
        x = torch.cat((x, observations[:, -self.theta_dim:]), dim=1) ## Concatenate along the last dimension to get a tensor of shape (batch_size, n_heads*n_odor + theta_dim)
        return x


def train_model(config):
    print(config)
    seed = int(sys.argv[1])
    rng = np.random.default_rng(seed)
    environment = HistoryNavigator(rng = rng, config = config)
    print(hasattr(environment, 'seed'))
    store_config(config)
    training_dict = config['training']
    policy_kwargs={"net_arch": [training_dict['N_HIDDEN_UNITS']]*training_dict['N_HIDDEN_LAYERS']}
    if training_dict['FEATURES_EXTRACTOR_CLASS'] is not None:
        policy_kwargs['features_extractor_class'] = training_dict['FEATURES_EXTRACTOR_CLASS']
        policy_kwargs['features_extractor_kwargs'] = { "n_odor": len(config['state']["FEATURES"]), "n_heads": training_dict['N_HEADS'], "history_len": config["state"]['HIST_LEN'], "theta_dim": environment.theta_dim}
    learning_rate = stable_baselines3.common.utils.get_linear_fn(start = training_dict['MAX_ALPHA'], end = training_dict['MIN_ALPHA'], end_fraction = training_dict['LEARNING_END_FRACTION'])
    model = DQN("MlpPolicy", environment, verbose = 1, tensorboard_log=training_dict["TB_LOG"], gamma = training_dict['GAMMA'], 
	exploration_final_eps = training_dict['MIN_EPSILON'], learning_rate=learning_rate, policy_kwargs=policy_kwargs,)
    model.learn(total_timesteps=training_dict['N_EPISODES']*training_dict['MAX_EPISODE_LENGTH'], tb_log_name=training_dict['MODEL_NAME'])
    # Save the model
    model.save(os.path.join(config["output"]["SAVE_DIRECTORY"], training_dict['MODEL_NAME']))
    environment.close()
    ## Free up memory - hope this does the job
    environment.close()
    del environment.odor_plume
    del environment
    return model

def test_model(config):
    config["output"]["RECORD_SUCCESS"] = True
    model = DQN.load(os.path.join(config["output"]["SAVE_DIRECTORY"], config["training"]['MODEL_NAME']))
    rng = np.random.default_rng(seed=1)
    render_env = HistoryNavigator(rng, config)
    obs = render_env.reset()
    episode_no = 0
    num_record = config["output"]["RECORD_STATE_ACTION"]
    state_arr = np.empty((num_record, config["plume"]["STOP_FRAME"], render_env.obs_dim))
    action_arr = np.empty((num_record, config["plume"]["STOP_FRAME"]))
    while episode_no < config["training"]['TEST_EPISODES']:
        action = model.predict(obs, deterministic=True)[0]
        if episode_no < num_record:
            ## Store state and action
            state_arr[episode_no, render_env.odor_plume.frame_number, :] = obs
            action_arr[episode_no, render_env.odor_plume.frame_number] = action
        obs, _, done, _ = render_env.step(action)
        if episode_no < 10:
            render_env.render()
        if done:
            obs = render_env.reset()
            episode_no += 1
            if episode_no % 100 == 0:
                print("Episode number: ", episode_no, flush=True)
    np.save(os.path.join(config["output"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_reward_history.npy"), np.array(render_env.all_episode_rewards))
    np.save(os.path.join(config["output"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_success_history.npy"), np.array(render_env.all_episode_success))
    ## Save state and action arrays
    np.save(os.path.join(config["output"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_state_history.npy"), state_arr)
    np.save(os.path.join(config["output"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_action_history.npy"), action_arr)
    print("Average reward: ", np.mean(render_env.all_episode_rewards))
    print("Average success: ", np.mean(render_env.all_episode_success))
    render_env.close()






