from sb3_contrib import RecurrentPPO
from src.environment.gym_environment_class import *


def train_model(config):
    training_dict = config['training']
    ## Define the environment here
    rng = np.random.default_rng(seed=0)
    environment = FlyNavigator(rng, config)
    ## Define the model to be run
    model_class = training_dict["model_class"]
    model = model_class(training_dict["policy"], environment, verbose=1, n_steps=training_dict["n_steps"], batch_size=training_dict["n_steps"]*training_dict["n_envs"], 
    policy_kwargs={"lstm_hidden_size": training_dict['lstm_hidden_size'], "net_arch": training_dict['actor_critic_layers']},
    gamma=training_dict['gamma'], gae_lambda=training_dict['gae_lambda'], clip_range=training_dict['clip_range'], vf_coef=training_dict['vf_coef'], ent_coef=training_dict['ent_coef'])
    # Train the model
    model.learn(total_timesteps=training_dict['n_episodes']*training_dict['max_episode_length'])
    # Save the model
    model.save(os.path.join('.', 'src', 'trained_models', training_dict['model_name']))
    return model