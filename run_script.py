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

config_dict['N_EPISODES'] = 2
config_dict['MOVIE_PATH'] = plume_movie_path
config_dict['MIN_FRAME'] = 500
config_dict['STOP_FRAME'] = 510
config_dict['RESET_FRAME_RANGE'] = np.array([501,509])


environment = PlumeMotionNavigationEnvironmentMovie1PlumeSourceRewardStopActionFactory(config=config_dict, actions=WalkStopActionEnum, rng = rng).plume_environment
environment.action_space.np_random.seed(seed)
#gym.utils.seeding.np_random(seed)

q_shape = np.append(environment.observation_space.nvec,environment.action_space.n)
q_table = np.zeros(shape=q_shape)
#print(q_shape)

epsilon = config_dict['MAX_EPSILON']
alpha = config_dict['MAX_ALPHA']
#alpha = MAX_ALPHA
#rng = np.random.default_rng(seed=12345)
min_reset_x = config_dict['MIN_RESET_X_MM']
max_reset_x = min_reset_x + config_dict['GOAL_RADIUS_MM']
print(f'starting less than {max_reset_x} away in x coordinate')

#reset_y_radius = 
transition_incrementer = 0
parameter_decay = 0
#log_interval = 50
recent_rewards = deque(maxlen=50)
#vis = visdom.Visdom()

# MAIN TRAINING LOOP

all_Q_shape = np.append(q_shape, config_dict['N_EPISODES'])

all_Q = np.zeros(all_Q_shape)

for episode in range(config_dict['N_EPISODES']):

    print('starting episode')
    #print('latest verion')

    if (transition_incrementer > 0) & (transition_incrementer % 150 == 0):
        old_reset_x = max_reset_x
        old_reset_y = reset_y_radius
        max_reset_x = np.min([config_dict['MAX_RESET_X_MM'],10*GOAL_RADIUS+max_reset_x])


    flip = environment.rng.choice([True,False],1)
    observation = environment.reset(options={'randomization_x_bounds':np.array([min_reset_x,max_reset_x]),
                                             'randomization_y_bounds': np.array([config_dict['MIN_RESET_Y_MM'], config_dict['MAX_RESET_Y_MM']]),
                                             'flip':flip})

    #print('done reset')
    done = False
 
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

        print("old frame = ", environment.odor_plume.frame_number)
        print("old position = ", environment.fly_spatial_parameters.position)
        print('action in run script = ', action)
        new_observation, reward, done, odor_measures = environment.step(action)
        print("new position = ", environment.fly_spatial_parameters.position)
        print("new_frame = ", environment.odor_plume.frame_number)

        #print('done =', done)
        

        if reward > 0:
            #print('received reward')
            total_rewards += 1

            parameter_decay +=1

        if reward < 0:
            parameter_decay +=1
            #print('hit a wall')




        update_index = tuple(np.append(observation,action))

        t1_value_index = tuple(new_observation)# Note the use of this index requires actions to be last axes of q table
        q_table[update_index] = \
            q_table[update_index] +\
            alpha * (reward + config_dict['GAMMA']*np.max(q_table[t1_value_index]) -\
            q_table[update_index])
        observation = new_observation

        epsilon = config_dict['MIN_EPSILON'] + (config_dict['MAX_EPSILON']-config_dict['MIN_EPSILON'])*np.exp(-config_dict['DECAY']*parameter_decay)
        alpha = config_dict['MIN_ALPHA'] + (config_dict['MAX_ALPHA']-config_dict['MIN_ALPHA']) * np.exp(-config_dict['DECAY']*parameter_decay)

    recent_rewards.append(reward)
    transition_incrementer += 1

    all_Q[...,episode] = q_table 

    print('done episode')
    
	#np.save(str(seed)+"_all_Q", all_Q)   
