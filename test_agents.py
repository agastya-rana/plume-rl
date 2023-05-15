from src.models.action_definitions import WalkStopActionEnum
from src.models.base_config import config_dict
import os
import sys 
from collections import deque

#sys.path.append('../')

import numpy as np
import gym
#import visdom
#import tqdm.notebook

#from src.models.goals import GOAL_X,GOAL_Y,GOAL_RADIUS
#from src.models.motion_environment_factory import PlumeMotionNavigationEnvironmentMovie1PlumeSourceRewardStopActionFactory
from src.models.motion_environment_factory import PlumeMotionNavigationEnvironmentMovie1PlumeSourceRewardStopActionFactory

seed = int(sys.argv[1])
rng = np.random.default_rng(seed)



"""
N_EPISODES = 2 # How many independently initialized runs to train on
MAX_ALPHA = 0.2 # Learning rate
MIN_ALPHA = 0.005
GAMMA = 0.95 # Reward temporal discount factor
MAX_EPSILON = 1 # Starting exploration rate
MIN_EPSILON = 0.01 # Asymptote of decaying exploration rate
DECAY = 0.05 # Rate of exploration decay

MIN_RESET_X = GOAL_X + 10 + GOAL_RADIUS # Initialization condition
MAX_RESET_X = 1400 # Initialization condition
rewards = np.zeros(N_EPISODES)
total_rewards = 0
"""

plume_movie_path = os.path.join('..', 'src', 'data', 'plume_movies', 'intermittent_smoke.avi')

config_dict['N_EPISODES'] = 200
config_dict['MOVIE_PATH'] = plume_movie_path
config_dict['MIN_FRAME'] = 500
config_dict['STOP_FRAME'] = 5000
config_dict['RESET_FRAME_RANGE'] = np.array([501,800])


environment = PlumeMotionNavigationEnvironmentMovie1PlumeSourceRewardStopActionFactory(config=config_dict, actions=WalkStopActionEnum, rng = rng).plume_environment
environment.action_space.np_random.seed(seed)
#gym.utils.seeding.np_random(seed)

#q_shape = np.append(environment.observation_space.nvec,environment.action_space.n)
#q_table = np.zeros(shape=q_shape)
#print(q_shape)

q_table_path = '../trained_models/all_good_Q'

all_good_q = np.load(q_table_path)

print("All good q shape = ", np.shape(all_good_q))

q_table = all_good_q[...,-1]


epsilon = 0
alpha = 0
#alpha = MAX_ALPHA
#rng = np.random.default_rng(seed=12345)
min_reset_x = config_dict['MIN_RESET_X_MM']
max_reset_x = config_dict['MAX_RESET_X_MM']
#print(f'starting less than {max_reset_x} away in x coordinate')

#reset_y_radius = 
transition_incrementer = 0
parameter_decay = 0
#log_interval = 50
recent_rewards = deque(maxlen=50)
#vis = visdom.Visdom()

# MAIN TRAINING LOOP

#all_Q_shape = np.append(q_shape, config_dict['N_EPISODES'])
#all_Q = np.zeros(all_Q_shape)

all_reward = np.zeros(200)
all_ep_time = np.zeros(200)

for episode in range(config_dict['N_EPISODES']):

    print('starting episode')
    #print('latest verion')

    flip = environment.rng.choice([True,False],1)
    observation = environment.reset(options={'randomization_x_bounds':np.array([min_reset_x,max_reset_x]),
                                             'randomization_y_bounds': np.array([config_dict['MIN_RESET_Y_MM'], config_dict['MAX_RESET_Y_MM']]),
                                             'flip':flip})

    #print('done reset')
    done = False
 
    ep_time = 0
    while not done: # Advance the environment (e.g., the smoke plume updates and the agent walks a step)
        explore = environment.rng.uniform() < epsilon# Can pick all random numbers at start
        if explore:
            action = environment.action_space.sample()
            #print('in explore')
        else:
            best = np.argwhere(q_table[tuple(observation)] == np.amax(q_table[tuple(observation)]))
            action = environment.rng.choice(best)
            #print('exploit')

            #action = np.argmax(q_table[tuple(observation)])
            #action = np.unravel_index(action,shape=environment.action_space.n).squeeze()

        new_observation, reward, done, odor_measures = environment.step(action)

        #print('done =', done)
        

        if reward > 0:
            #print('received reward')
            total_rewards += 1

            parameter_decay +=1

        if reward < 0:
            parameter_decay +=1
            #print('hit a wall')


        observation = new_observation

        ep_time +=1

    recent_rewards.append(reward)
    transition_incrementer += 1

    print('reward received = ', reward)
    print('done episode')
    all_reward[episode] = reward
    all_ep_time[episode] = ep_time
    
np.save(str(seed)+"_all_reward", all_reward)
np.save(str(seed)+"_all_ep_time", all_ep_time)

