import numpy as np
from src.models.gym_environment_class import *
from src.models.base_config import *
import os

plume_movie_path = os.path.join('.', 'src', 'data', 'plume_movies', 'intermittent_smoke.avi')
config_dict['MOVIE_PATH'] = plume_movie_path
config_dict['MIN_FRAME'] = 500
config_dict['STOP_FRAME'] = 1000
config_dict['RESET_FRAME_RANGE'] = np.array([501,509])
config_dict['VIDEO'] = True
config_dict["MIN_TURN_DUR_S"] = 0.05
config_dict["EXCESS_TURN_DUR_S"] = 0.05
config_dict["CONCENTRATION_BASE_THRESHOLD"] = 20

rng = np.random.default_rng(seed=0)
env = FlyNavigator(rng, config_dict)
env.reset()
for episode in range(1):
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
    env.reset()
