from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import DQN, PPO
from sb3_contrib import RecurrentPPO
from src.environment.utilities import *
from src.models.callbacks import *
import stable_baselines3
import numpy as np
import os
import gym.spaces

class SB3Model():
    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.model_class = config["training"]["MODEL_CLASS"]
        self.env_class = config["training"]["ENV_CLASS"]

    def train(self):
        if self.model_class == "RecurrentPPO" or self.model_class == "PPO":
            n_envs = self.config["training"]["N_ENVS"] if "N_ENVS" in self.config["training"] else 1
            environment = VecMonitor(SubprocVecEnv([(lambda: self.env_class(self.rng, self.config)) for i in range(n_envs)]))
        else:
            environment = self.env_class(self.rng, self.config)
        store_config(self.config)
        config = self.config
        training_dict = config['training']
        callback_model = config['training']['CALLBACK'] if 'CALLBACK' in config['training'] else None
        if callback_model is not None:
            assert config["training"]["RECORD_SUCCESS"] == True, "If using a callback, RECORD_SUCCESS must be set to True"
            callback = callback_model(config)
    
        ## Train based on the model class
        if self.model_class == 'RecurrentPPO':
            model = self.make_RNN_model(environment)        
        elif self.model_class == 'PPO':
            model = self.make_PPO_model(environment)
        elif self.model_class == 'DQN':
            model = self.make_DQN_model(environment)
        else:
            raise NotImplementedError
        
        log_interval = training_dict.get('LOG_INTERVAL', 10)
        ## Train the model
        model.learn(total_timesteps=training_dict['TRAIN_TIMESTEPS'], tb_log_name=training_dict['MODEL_NAME'], callback=callback, log_interval=log_interval)
        # Save the model
        model.save(os.path.join(config["training"]["SAVE_DIRECTORY"], training_dict['MODEL_NAME']))
        ## Free up memory - hope this does the job
        environment.close()
        del environment  
        return model

    def make_RNN_model(self, environment):
        config = self.config
        training_dict = config['training']
        gamma = training_dict.get('GAMMA', 1)
        vf_coef = training_dict.get('VF_COEF', 0.5)
        ent_coef = training_dict.get('ENT_COEF', 0.01)
        clip_range = training_dict.get('CLIP_RANGE', 0.2)
        gae_lambda = training_dict.get('GAE_LAMBDA', 0.95)
        n_steps = training_dict.get('N_STEPS', 2048)
        n_epochs = training_dict.get('N_EPOCHS', 10)
        if 'N_HIDDEN_UNITS' in training_dict and 'N_HIDDEN_LAYERS' in training_dict:
            arch = [training_dict['N_HIDDEN_UNITS']]*training_dict['N_HIDDEN_LAYERS']
        elif 'ACTOR_CRITIC_LAYERS' in training_dict:
            arch = training_dict['ACTOR_CRITIC_LAYERS']
        else:
            arch = [64, 64]
        
        policy_kwargs = {"lstm_hidden_size": training_dict['LSTM_HIDDEN_SIZE'], "net_arch": arch}
        policy_kwargs = self._add_feature_extractor_policy(policy_kwargs, config, environment)
        learning_rate = self._get_learning_rate(training_dict)

        model = RecurrentPPO("MlpLstmPolicy", environment, verbose=1, n_steps=n_steps, batch_size=n_steps*training_dict["N_ENVS"], 
        policy_kwargs=policy_kwargs,
        gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, vf_coef=vf_coef, ent_coef=ent_coef,
        tensorboard_log=training_dict['TB_LOG'], learning_rate=learning_rate, n_epochs=n_epochs)
        return model

    def make_PPO_model(self, environment):
        config = self.config
        training_dict = config['training']
        gamma = training_dict.get('GAMMA', 1)
        vf_coef = training_dict.get('VF_COEF', 0.5)
        ent_coef = training_dict.get('ENT_COEF', 0.01)
        clip_range = training_dict.get('CLIP_RANGE', 0.2)
        gae_lambda = training_dict.get('GAE_LAMBDA', 0.95)
        n_steps = training_dict.get('N_STEPS', 2048)
        n_epochs = training_dict.get('N_EPOCHS', 10)

        if 'N_HIDDEN_UNITS' in training_dict and 'N_HIDDEN_LAYERS' in training_dict:
            arch = [training_dict['N_HIDDEN_UNITS']]*training_dict['N_HIDDEN_LAYERS']
        elif 'ACTOR_CRITIC_LAYERS' in training_dict:
            arch = training_dict['ACTOR_CRITIC_LAYERS']
        else:
            arch = [64, 64]
        
        policy_kwargs={"net_arch": {"pi": arch, "vf": arch}}
        policy_kwargs = self._add_feature_extractor_policy(policy_kwargs, config, environment)

        learning_rate = self._get_learning_rate(training_dict)

        model = PPO("MlpPolicy", environment, verbose = 1, tensorboard_log=training_dict["TB_LOG"], gamma = gamma, n_steps=n_steps, batch_size=n_steps*training_dict["N_ENVS"], n_epochs=n_epochs,
        gae_lambda=gae_lambda, clip_range=clip_range, vf_coef=vf_coef, ent_coef=ent_coef, learning_rate=learning_rate, policy_kwargs=policy_kwargs,)
        return model

    def make_DQN_model(self, environment):
        config = self.config
        training_dict = config['training']
        exploration_fraction = training_dict.get('EXPLORATION_FRACTION', 0.1)
        gamma = training_dict.get('GAMMA', 1)
        min_epsilon = training_dict.get('MIN_EPSILON', 0.01)
        target_update_interval = training_dict.get('TARGET_UPDATE_INTERVAL', 10000)

        if 'N_HIDDEN_UNITS' in training_dict and 'N_HIDDEN_LAYERS' in training_dict:
            arch = [training_dict['N_HIDDEN_UNITS']]*training_dict['N_HIDDEN_LAYERS']
        elif 'ACTOR_CRITIC_LAYERS' in training_dict:
            arch = training_dict['ACTOR_CRITIC_LAYERS']
        else:
            arch = [64, 64]
        
        policy_kwargs={"net_arch": arch}
        policy_kwargs = self._add_feature_extractor_policy(policy_kwargs, config, environment)
        
        learning_rate = self._get_learning_rate(training_dict)
        
        model = DQN("MlpPolicy", environment, verbose = 1, tensorboard_log=training_dict["TB_LOG"], gamma = gamma, 
        exploration_final_eps = min_epsilon, learning_rate=learning_rate, policy_kwargs=policy_kwargs, exploration_fraction=exploration_fraction, 
        target_update_interval=target_update_interval,)
        return model

    @staticmethod
    def _add_feature_extractor_policy(policy_kwargs, config, environment):
        training_dict = config['training']
        if training_dict['FEATURES_EXTRACTOR_CLASS'] is not None:
            policy_kwargs['features_extractor_class'] = training_dict['FEATURES_EXTRACTOR_CLASS']
            use_sin_cos = config["state"]['USE_COSINE_AND_SIN_THETA'] if 'USE_COSINE_AND_SIN_THETA' in config["state"] else True
            theta_dim = 2 if use_sin_cos else 1
            policy_kwargs['features_extractor_kwargs'] = { "n_odor": len(config['state']["FEATURES"]), "n_heads": training_dict['N_HEADS'], "history_len": config["state"]['HIST_LEN'], "theta_dim": theta_dim}
        return policy_kwargs

    @staticmethod
    def _get_learning_rate(training_dict):
        min_alpha = training_dict.get('MIN_ALPHA', 0.0001)
        max_alpha = training_dict.get('MAX_ALPHA', min_alpha)
        learning_end_fraction = training_dict.get('LEARNING_END_FRACTION', 0.1)
        learning_rate = stable_baselines3.common.utils.get_linear_fn(start=max_alpha, end=min_alpha, end_fraction=learning_end_fraction)
        return learning_rate

    @staticmethod
    def _get_action_space_dim(env):
        if isinstance(env.action_space, gym.spaces.Box):
            return env.action_space.shape
        elif isinstance(env.action_space, gym.spaces.Discrete):
            return (1,)
        else:
            raise NotImplementedError
    
    def test(self):
        config = self.config
        config["training"]["RECORD_SUCCESS"] = True
        ## Reset max x to default
        if 'INITIAL_MAX_RESET_X_MM' in config['plume']:
            del config['plume']['INITIAL_MAX_RESET_X_MM']
        if 'PLUME_DICT_LIST' in config['plume']:
            max_frames = 0
            for plume_dict in config['plume']['PLUME_DICT_LIST']:
                max_frames = max(max_frames, plume_dict['STOP_FRAME'])
                if 'INITIAL_MAX_RESET_X_MM' in plume_dict:
                    del plume_dict['INITIAL_MAX_RESET_X_MM']
        else:
            max_frames = config['plume']['STOP_FRAME']
        
        ## Load the model
        if self.model_class == "RecurrentPPO":
            model = RecurrentPPO.load(os.path.join(config["training"]["SAVE_DIRECTORY"], config["training"]['MODEL_NAME']))
        elif self.model_class == "PPO":
            model = PPO.load(os.path.join(config["training"]["SAVE_DIRECTORY"], config["training"]['MODEL_NAME']))
        elif self.model_class == "DQN":
            model = DQN.load(os.path.join(config["training"]["SAVE_DIRECTORY"], config["training"]['MODEL_NAME']))
        test_env = self.env_class(self.rng, config)
        obs = test_env.reset()
        if self.model_class == "RecurrentPPO":
            # cell and hidden state of the LSTM
            lstm_states = None
            # Episode start signal
            episode_start = True
        ## Initialize relevant arrays
        episode_no = 0
        num_record = config["training"]["RECORD_STATE_ACTION"]
        num_render = config["training"]["RECORD_RENDER"] if "RECORD_RENDER" in config["training"] else 10
        state_arr = np.empty((num_record, max_frames, test_env.obs_dim))
        action_arr = np.empty((num_record, max_frames,) + self._get_action_space_dim(test_env))

        ## Run the test
        while episode_no < config["training"]['TEST_EPISODES']:
            if self.model_class == "RecurrentPPO":
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
            elif self.model_class == "RecurrentPPO":
                episode_start = False
        
        ## Save reward and success histories
        np.save(os.path.join(config["training"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_reward_history.npy"), np.array(test_env.all_episode_rewards))
        np.save(os.path.join(config["training"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_success_history.npy"), np.array(test_env.all_episode_success))
        ## Save state and action arrays
        np.save(os.path.join(config["training"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_state_history.npy"), state_arr)
        np.save(os.path.join(config["training"]["SAVE_DIRECTORY"],config["training"]["MODEL_NAME"]+"_action_history.npy"), action_arr)
        print("Average reward: ", np.mean(test_env.all_episode_rewards))
        print("Average success: ", np.mean(test_env.all_episode_success))
        r, s = test_env.all_episode_rewards, test_env.all_episode_success
        test_env.close()
        del test_env
        return r, s