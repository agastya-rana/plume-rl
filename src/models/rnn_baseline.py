from sb3_contrib import RecurrentPPO
from src.environment.gym_environment_class import *
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import os
import numpy as np

# Helper function to create environments
def make_env(i, config_dict):
    def _init():
        return FlyNavigator(np.random.default_rng(seed=i), config_dict)
    return _init


def train_model(config):
    training_dict = config['training']
    ## Define the environment here
    rng = np.random.default_rng(seed=0)
    environment = FlyNavigator(rng, config)
    ## Define the model to be run
    model_class = training_dict["MODEL_CLASS"]
    # Create vectorized environments
    env = SubprocVecEnv([make_env(i, config) for i in range(training_dict["N_ENVS"])])
    env = VecMonitor(env)#, info_keywords=('ep_rew_mean', 'ep_len_mean')); can add these if add corresponding params to info dict in step function of environment
    model = model_class(training_dict["POLICY"], env, verbose=1, n_steps=training_dict["N_STEPS"], batch_size=training_dict["N_STEPS"]*training_dict["N_ENVS"], 
    policy_kwargs={"lstm_hidden_size": training_dict['LSTM_HIDDEN_SIZE'], "net_arch": training_dict['ACTOR_CRITIC_LAYERS']},
    gamma=training_dict['GAMMA'], gae_lambda=training_dict['GAE_LAMBDA'], clip_range=training_dict['CLIP_RANGE'], vf_coef=training_dict['VF_COEF'], ent_coef=training_dict['ENT_COEF'],
    tensorboard_log=training_dict['TB_LOG'])
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
    ## TODO: Store the state variables and actions per timestep in an array. Also, only render for the first 10 episodes.
    while episode_no < config["training"]['TEST_EPISODES']:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
        obs, _, done, _ = render_env.step(action)
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
    print("Average reward: ", np.mean(render_env.all_episode_rewards))
    print("Average success: ", np.mean(render_env.all_episode_success))
    render_env.close()