
import torch.nn as nn
import torch.nn.functional as F
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import DQN, PPO
from sb3_contrib import RecurrentPPO
from src.environment.utilities import *
from src.models.callbacks import *
import stable_baselines3
import numpy as np
import os
import sys
import gym.spaces as spaces

class SB3Model():
    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.model_class = config["training"]["MODEL_CLASS"]
        self.env_class = config["training"]["ENV_CLASS"]
    
    def make_env(self):
        if self.model_class == RecurrentPPO:
            n_envs = self.config["training"]["N_ENVS"] if "N_ENVS" in self.config["training"] else 1
            return VecMonitor(SubprocVecEnv([self.env_class(self.rng, self.config) for i in range(n_envs)]))
        else:
            return self.env_class(self.rng, self.config)
    
    ## change log interval here too
    def train(self):
        environment = self.make_env()
        store_config(self.config)
        training_dict = config['training']
        config = self.config
        callback_model = config['training']['CALLBACK'] if 'CALLBACK' in config['training'] else None
        if callback_model is not None:
            assert config["training"]["RECORD_SUCCESS"] == True, "If using a callback, RECORD_SUCCESS must be set to True"
            callback = callback_model(config)
        ## Train based on the model class
        if self.model_class == RecurrentPPO:
            model = self.model_class("MlpPolicy", environment, verbose=1, n_steps=training_dict["N_STEPS"], batch_size=training_dict["N_STEPS"]*training_dict["N_ENVS"], 
            policy_kwargs={"lstm_hidden_size": training_dict['LSTM_HIDDEN_SIZE'], "net_arch": training_dict['ACTOR_CRITIC_LAYERS']},
            gamma=training_dict['GAMMA'], gae_lambda=training_dict['GAE_LAMBDA'], clip_range=training_dict['CLIP_RANGE'], vf_coef=training_dict['VF_COEF'], ent_coef=training_dict['ENT_COEF'],
            tensorboard_log=training_dict['TB_LOG'], learning_rate=training_dict['LEARNING_RATE'])
            # Train the model
            model.learn(total_timesteps=training_dict['N_EPISODES']*training_dict['MAX_EPISODE_LENGTH'], tb_log_name=training_dict['MODEL_NAME'], callback=callback)
        
        elif self.model_class == PPO:
            arch = [training_dict['N_HIDDEN_UNITS']]*training_dict['N_HIDDEN_LAYERS']
            policy_kwargs={"net_arch": {"pi": arch, "vf": arch}}
            if training_dict['FEATURES_EXTRACTOR_CLASS'] is not None:
                policy_kwargs['features_extractor_class'] = training_dict['FEATURES_EXTRACTOR_CLASS']
                policy_kwargs['features_extractor_kwargs'] = { "n_odor": len(config['state']["FEATURES"]), "n_heads": training_dict['N_HEADS'], "history_len": config["state"]['HIST_LEN'], "theta_dim": environment.theta_dim}
            learning_rate = stable_baselines3.common.utils.get_linear_fn(start = training_dict['MAX_ALPHA'], end = training_dict['MIN_ALPHA'], end_fraction = training_dict['LEARNING_END_FRACTION'])
            model = self.model_class("MlpPolicy", environment, verbose = 1, tensorboard_log=training_dict["TB_LOG"], gamma = training_dict['GAMMA'], 
            n_steps=training_dict["N_STEPS"], batch_size=training_dict["N_STEPS"]*training_dict["N_ENVS"], n_epochs=training_dict["N_EPOCHS"],
            gae_lambda=training_dict['GAE_LAMBDA'], clip_range=training_dict['CLIP_RANGE'], vf_coef=training_dict['VF_COEF'], ent_coef=training_dict['ENT_COEF'],
            learning_rate=learning_rate, policy_kwargs=policy_kwargs,)
            model.learn(total_timesteps=training_dict['N_EPISODES']*training_dict['MAX_EPISODE_LENGTH'], tb_log_name=training_dict['MODEL_NAME'], callback=callback)
        
        elif self.model_class == DQN:
            exploration_fraction = training_dict['EXPLORATION_FRACTION'] if 'EXPLORATION_FRACTION' in training_dict else 0.1
            policy_kwargs = {"net_arch": [training_dict['N_HIDDEN_UNITS']]*training_dict['N_HIDDEN_LAYERS']}        
            if training_dict['FEATURES_EXTRACTOR_CLASS'] is not None:
                policy_kwargs['features_extractor_class'] = training_dict['FEATURES_EXTRACTOR_CLASS']
                policy_kwargs['features_extractor_kwargs'] = { "n_odor": len(config['state']["FEATURES"]), "n_heads": training_dict['N_HEADS'], "history_len": config["state"]['HIST_LEN'], "theta_dim": environment.theta_dim}
            
            learning_rate = stable_baselines3.common.utils.get_linear_fn(start = training_dict['MAX_ALPHA'], end = training_dict['MIN_ALPHA'], end_fraction = training_dict['LEARNING_END_FRACTION'])
            
            model = self.model_class("MlpPolicy", environment, verbose = 1, tensorboard_log=training_dict["TB_LOG"], gamma = training_dict['GAMMA'], 
            exploration_final_eps = training_dict['MIN_EPSILON'], learning_rate=learning_rate, policy_kwargs=policy_kwargs, exploration_fraction=exploration_fraction, 
            target_update_interval=training_dict['TARGET_UPDATE_INTERVAL'],)
            model.learn(total_timesteps=training_dict['N_EPISODES']*training_dict['MAX_EPISODE_LENGTH'], tb_log_name=training_dict['MODEL_NAME'], callback=callback)
            
        else:
            raise NotImplementedError
            
        # Save the model
        model.save(os.path.join(config["training"]["SAVE_DIRECTORY"], training_dict['MODEL_NAME']))
        ## Free up memory - hope this does the job
        environment.close()
        del environment        
        return model

    @staticmethod
    def _get_action_space_dim(env):
        if isinstance(env.action_space, gym.spaces.Box):
            return env.action_space.shape
        elif isinstance(env.action_space, gym.spaces.Discrete):
            return (1,)
        else:
            raise NotImplementedError
    
    def test_model():
        config = self.config
        config["training"]["RECORD_SUCCESS"] = True
        ## Reset max x to default
        if 'INITIAL_MAX_RESET_X_MM' in config['plume']:
            del config['plume']['INITIAL_MAX_RESET_X_MM']
        elif 'PLUME_DICT_LIST' in config['plume']:
            for plume_dict in config['plume']['PLUME_DICT_LIST']:
                if 'INITIAL_MAX_RESET_X_MM' in plume_dict:
                    del plume_dict['INITIAL_MAX_RESET_X_MM']
        model = self.model_class.load(os.path.join(config["training"]["SAVE_DIRECTORY"], config["training"]['MODEL_NAME']))
        test_env = FlyNavigator(self.rng, config)
        obs = test_env.reset()
        if self.model_class == RecurrentPPO:
            # cell and hidden state of the LSTM
            lstm_states = None
            # Episode start signal
            episode_start = True
        ## Initialize relevant arrays
        episode_no = 0
        num_record = config["training"]["RECORD_STATE_ACTION"]
        num_render = config["training"]["RECORD_RENDER"] if "RECORD_RENDER" in config["training"] else 10
        state_arr = np.empty((num_record, config["plume"]["STOP_FRAME"], test_env.obs_dim))
        action_arr = np.empty((num_record, config["plume"]["STOP_FRAME"],) + _get_action_space_dim(test_env))

        ## Run the test
        while episode_no < config["training"]['TEST_EPISODES']:
            if self.model_class == RecurrentPPO:
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            if episode_no < num_record:
                ## Store state and action
                state_arr[episode_no, test_env.odor_plume.frame_number, :] = obs
                action_arr[episode_no, test_env.odor_plume.frame_number, :] = action
            obs, _, done, _ = test_env.step(action)
            if episode_no < num_render:
                test_env.render()
            # If the episode is done, reset the environment (for vector environments, we don't need to reset manually, but assume it's not vector)
            if done:
                obs = test_env.reset()
                episode_no += 1
                episode_start = True
                if episode_no % 100 == 0:
                    print("Episode ", episode_no)
            elif self.model_class == RecurrentPPO:
                episode_start = False
        
        ## Save reward and success histories
        np.save(os.path.join(config["training"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_reward_history.npy"), np.array(test_env.all_episode_rewards))
        np.save(os.path.join(config["training"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_success_history.npy"), np.array(test_env.all_episode_success))
        ## Save state and action arrays
        np.save(os.path.join(config["training"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_state_history.npy"), state_arr)
        np.save(os.path.join(config["training"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_action_history.npy"), action_arr)
        print("Average reward: ", np.mean(test_env.all_episode_rewards))
        print("Average success: ", np.mean(test_env.all_episode_success))
        test_env.close()
        
