from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import DQN, PPO
from sb3_contrib import RecurrentPPO
from src.environment.utilities import *
from src.models.callbacks import *
import stable_baselines3
import numpy as np
import time
import os
import gym.spaces

class SB3Model():
    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.model_class = config["training"]["MODEL_CLASS"]
        self.env_class = config["training"]["ENV_CLASS"]
        self.model = None

    def train(self, model_path=None, load_model=False):
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

        if not load_model:
            ## Train based on the model class
            if self.model_class == 'RecurrentPPO':
                model = self.make_RNN_model(environment)
            elif self.model_class == 'PPO':
                model = self.make_PPO_model(environment)
            elif self.model_class == 'DQN':
                model = self.make_DQN_model(environment)
            else:
                raise NotImplementedError
        else:
            if model_path is None:
                model_path = os.path.join(config["training"]["SAVE_DIRECTORY"], config["training"]['MODEL_NAME'])
            print("Loading existing model")
            model = self.load_model(model_path, environment)
        
        log_interval = training_dict.get('LOG_INTERVAL', 10)
        ## Train the model
        model.learn(total_timesteps=training_dict['TRAIN_TIMESTEPS'], tb_log_name=training_dict['MODEL_NAME'], callback=callback, log_interval=log_interval)
        # Save the model
        model.save(os.path.join(config["training"]["SAVE_DIRECTORY"], training_dict['MODEL_NAME']))
        print("Model saved", flush=True)
        self.model = model
        print("Training complete", flush=True)
        return None

    def load_model(self, model_path, environment=None):
        if self.model_class == 'RecurrentPPO':
            model = RecurrentPPO.load(model_path, env=environment)
        elif self.model_class == 'PPO':
            model = PPO.load(model_path, env=environment)
        elif self.model_class == 'DQN':
            model = DQN.load(model_path, env=environment)
        else:
            raise NotImplementedError
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
    def _get_action_space_dim(action_space):
        if isinstance(action_space, gym.spaces.Box):
            return action_space.shape
        elif isinstance(action_space, gym.spaces.Discrete):
            return (1,)
        else:
            raise NotImplementedError
    
    def _find_max_max_frames(self):
        config = self.config
        if 'PLUME_DICT_LIST' in config['plume']:
            max_frames = 0
            for plume_dict in config['plume']['PLUME_DICT_LIST']:
                max_frames = max(max_frames, plume_dict['STOP_FRAME'])
        else:
            max_frames = config['plume']['STOP_FRAME']
        return max_frames

    def test(self, best=False):
        print("Testing", flush=True)
        config = self.config
        save_path = os.path.join(config["training"]["SAVE_DIRECTORY"], config["training"]['MODEL_NAME'])
        ## Don't need to reset max x reset since this would be annealed during training
        ## Load the model
        if self.model is None:
            print("Loading model for testing since it was not loaded before", flush=True)
            if best:
                self.model = self.load_model(save_path + '_best')
            else:
                self.model = self.load_model(save_path)
        else:
            print("Using model that was loaded before", flush=True)
        
        ## Is the problem that the model also stores the environment?
        print("Model loaded successfully; starting test")
        print(type(self.model))
        ## Initialize relevant arrays
        episode_no = 0
        num_render = config["training"]["RENDER_TIMESTEPS"] if "RENDER_TIMESTEPS" in config["training"] else 10000
        total_timesteps = config["training"]['TEST_TIMESTEPS']
        record_steps = config["training"]['RECORD_TIMESTEPS'] if 'RECORD_TIMESTEPS' in config["training"] else total_timesteps
        
        env = self.model.get_env()
        num_envs = env.num_envs
        state_arr = np.empty((num_envs, record_steps, env.get_attr("obs_dim")[0]))
        action_arr = np.empty((num_envs, record_steps,) + self._get_action_space_dim(env.action_space))
        reward_arr = np.empty((num_envs, record_steps))
        state = None
        dones = np.ones((num_envs,), dtype=bool)
        timestep = 0
        ## Run the test
        print(time.time(), flush=True)
        obs = env.reset()
        while timestep < total_timesteps:
            action, state = self.model.predict(obs, state=state, episode_start=dones, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            if timestep < record_steps:
                state_arr[:, timestep, :] = obs
                action_arr[:, timestep, :] = action.reshape(num_envs, -1)
                reward_arr[:, timestep] = rewards
            if timestep < num_render:
                env.render()
            timestep += 1
            if timestep % 1000 == 0:
                print("Timestep: ", timestep, flush=True)
        print(time.time(), flush=True)
        all_episode_rewards = np.array(env.get_attr("all_episode_rewards"))
        all_episode_success = np.array(env.get_attr("all_episode_success"))
        ## Save reward and success histories
        np.save(save_path + "_reward_history.npy", all_episode_rewards)
        np.save(save_path + "_success_history.npy", all_episode_success)
        np.save(save_path + "_state_history.npy", state_arr)
        np.save(save_path + "_action_history.npy", action_arr)
        ## Save state and action arrays
        print("Average reward: ", np.mean(all_episode_rewards))
        print("Average success: ", np.mean(all_episode_success))
        env.close()
        return all_episode_rewards, all_episode_success
    
    def test_new_env(self, best=False):
        print("Testing", flush=True)
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
        if self.model is None:
            print("Loading model for testing since it was not loaded before", flush=True)
            if best:
                self.model = self.load_model(os.path.join(config["training"]["SAVE_DIRECTORY"], config["training"]['MODEL_NAME'] + '_best'))
            else:
                self.model = self.load_model(os.path.join(config["training"]["SAVE_DIRECTORY"], config["training"]['MODEL_NAME']))
        else:
            print("Using model that was loaded before", flush=True)
            model = self.model
        
        ## Is the problem that the model also stores the environment?
        print("Model loaded successfully; starting test")
        ## Change this to work with VecEnv...
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
                action, lstm_states = self.model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            else:
                action, _ = self.model.predict(obs, deterministic=True)
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