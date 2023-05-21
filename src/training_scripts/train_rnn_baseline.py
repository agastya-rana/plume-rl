## Trains the baseline RNN on the odor plume using PPO on actor-critic MLP heads stemming from the RNN feature extractor
from src.models.rnn_baseline import *
from src.models.gym_environment_class import *
from src.models.base_config import *
import os
import numpy as np
plume_movie_path = os.path.join('.', 'src', 'data', 'plume_movies', 'intermittent_smoke.avi')

config_dict['N_EPISODES'] = 2
config_dict['MOVIE_PATH'] = plume_movie_path
config_dict['MIN_FRAME'] = 500
config_dict['STOP_FRAME'] = 510
config_dict['RESET_FRAME_RANGE'] = np.array([501,509])

## Define the environment here
rng = np.random.default_rng(seed=0)
environment = FlyNavigator(rng, config_dict)
## Define the model to be run
model = define_model(environment, config_dict)
# Train the model
model.learn(total_timesteps=10000)
# Save the model
model.save("rnn_baseline")
# Load the model
model = PPO.load("rnn_baseline")
# Evaluate the model
obs = env.reset()
## Check to see how done is evaluated and copy training metrics from other places
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
env.close()
