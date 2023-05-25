from sb3_contrib import RecurrentPPO
from src.environment.gym_environment_class import *
from stable_baselines3.common.vec_env import SubprocVecEnv

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
    ## TODO: add functionality to input a vector environment; see if you can use SubprocVecEnv here
    model = model_class(training_dict["policy"], env, verbose=1, n_steps=training_dict["n_steps"], batch_size=training_dict["n_steps"]*training_dict["n_envs"], 
    policy_kwargs={"lstm_hidden_size": training_dict['lstm_hidden_size'], "net_arch": training_dict['actor_critic_layers']},
    gamma=training_dict['gamma'], gae_lambda=training_dict['gae_lambda'], clip_range=training_dict['clip_range'], vf_coef=training_dict['vf_coef'], ent_coef=training_dict['ent_coef'],
    tensorboard_log=training_dict['tensorboard_log'])
    # Train the model

    model.learn(total_timesteps=training_dict['n_episodes']*training_dict['max_episode_length'])
    # Save the model
    model.save(os.path.join('.', 'src', 'trained_models', training_dict['model_name']))
    return model