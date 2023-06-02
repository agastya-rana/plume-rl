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
    model_class = training_dict["model_class"]
    # Create vectorized environments
    env = SubprocVecEnv([make_env(i, config) for i in range(training_dict["n_envs"])])
    env = VecMonitor(env)#, info_keywords=('ep_rew_mean', 'ep_len_mean')); can add these if add corresponding params to info dict in step function of environment
    model = model_class(training_dict["policy"], env, verbose=1, n_steps=training_dict["n_steps"], batch_size=training_dict["n_steps"]*training_dict["n_envs"], 
    policy_kwargs={"lstm_hidden_size": training_dict['lstm_hidden_size'], "net_arch": training_dict['actor_critic_layers']},
    gamma=training_dict['gamma'], gae_lambda=training_dict['gae_lambda'], clip_range=training_dict['clip_range'], vf_coef=training_dict['vf_coef'], ent_coef=training_dict['ent_coef'],
    tensorboard_log=training_dict['tensorboard_log'])
    # Train the model

    model.learn(total_timesteps=training_dict['n_episodes']*training_dict['max_episode_length'], tb_log_name=training_dict['model_name'])
    # Save the model
    model.save(os.path.join('.', 'src', 'trained_models', training_dict['model_name']))
    return model

def test_model(model, config):
    rng = np.random.default_rng(seed=1)
    render_env = FlyNavigator(rng, config)
    obs = render_env.reset()
    # cell and hidden state of the LSTM
    lstm_states = None
    # Episode start signal
    episode_start = True
    for st in range(4000):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
        obs, _, done, _ = render_env.step(action)
        render_env.render()
        # If the episode is done, reset the environment (for vector environments, we don't need to reset manually)
        if done:
            obs = render_env.reset()
            episode_start = True
        else:
            episode_start = False
    render_env.close()