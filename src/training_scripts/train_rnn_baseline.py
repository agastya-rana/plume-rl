## Trains the baseline RNN on the odor plume using PPO on actor-critic MLP heads stemming from the RNN feature extractor
from src.models.rnn_baseline import *
from src.models.gym_environment_class import *
from src.models.base_config import *
import os
import numpy as np
plume_movie_path = os.path.join('.', 'src', 'data', 'plume_movies', 'intermittent_smoke.avi')

config_dict['N_EPISODES'] = 1000000
config_dict['MOVIE_PATH'] = plume_movie_path
config_dict['MIN_FRAME'] = 500
config_dict['STOP_FRAME'] = 550
config_dict['RESET_FRAME_RANGE'] = np.array([501,509])
config_dict['RNN_HIDDEN_SIZE'] = 128

## Define the environment here
rng = np.random.default_rng(seed=0)
environment = FlyNavigator(rng, config_dict)
## Define the model to be run
model = RecurrentPPO("MlpLstmPolicy", environment, verbose=1, n_steps=128, batch_size=128*8, policy_kwargs={"lstm_hidden_size": config_dict['RNN_HIDDEN_SIZE']})
# Train the model
model.learn(total_timesteps=100000)
# Save the model
model.save("ppo_recurrent")

model = RecurrentPPO.load("ppo_recurrent")
obs = vec_env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    episode_starts = dones
    vec_env.render("human")