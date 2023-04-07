from src.models.action_definitions import WalkStopActionEnum
import os
import sys 
from collections import deque

#sys.path.append('../')

import numpy as np
#import visdom
#import tqdm.notebook

from src.models.goals import GOAL_X,GOAL_Y,GOAL_RADIUS
#from src.models.motion_environment_factory import PlumeMotionNavigationEnvironmentMovie1PlumeSourceRewardStopActionFactory
from src.models.motion_environment_factory import PlumeMotionNavigationEnvironment_motion_knockout_Movie1PlumeSourceRewardStopActionFactory

seed = int(sys.argv[1])

N_EPISODES = 2000 # How many independently initialized runs to train on
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


plume_movie_path = os.path.join('..', 'src', 'data', 'plume_movies', 'intermittent_smoke.avi')
environment = PlumeMotionNavigationEnvironment_motion_knockout_Movie1PlumeSourceRewardStopActionFactory(movie_file_path=plume_movie_path,actions=WalkStopActionEnum).plume_environment
q_shape = np.append(environment.observation_space.nvec,environment.action_space.n)
q_table = np.zeros(shape=q_shape)
#print(q_shape)

epsilon = MAX_EPSILON
alpha = MAX_ALPHA
rng = np.random.default_rng(seed=12345)
min_reset_x = MIN_RESET_X
max_reset_x = min_reset_x + GOAL_RADIUS
print(f'starting less than {max_reset_x} away in x coordinate')
reset_y_radius = 400
transition_incrementer = 0
parameter_decay = 0
log_interval = 50
recent_rewards = deque(maxlen=50)
#vis = visdom.Visdom()

# MAIN TRAINING LOOP

all_Q_shape = np.append(q_shape, N_EPISODES)

all_Q = np.zeros(all_Q_shape)

for episode in range(N_EPISODES):

    if (transition_incrementer > 0) & (transition_incrementer % 150 == 0):
        old_reset_x = max_reset_x
        old_reset_y = reset_y_radius
        max_reset_x = np.min([MAX_RESET_X,10*GOAL_RADIUS+max_reset_x])


    flip = np.random.choice([True,False],1)
    observation = environment.reset(options={'randomization_x_bounds':np.array([min_reset_x,max_reset_x]),
                                             'randomization_y_bounds': np.array([-reset_y_radius, reset_y_radius]) + GOAL_Y,
                                             'flip':flip})

    done = False
 
    while not done: # Advance the environment (e.g., the smoke plume updates and the agent walks a step)
        explore = rng.random() < epsilon# Can pick all random numbers at start
        if explore:
            action = environment.action_space.sample()
        else:
            best = np.argwhere(q_table[tuple(observation)] == np.amax(q_table[tuple(observation)]))
            action = rng.choice(best)

            #action = np.argmax(q_table[tuple(observation)])
            #action = np.unravel_index(action,shape=environment.action_space.n).squeeze()

        new_observation, reward, done, odor_measures = environment.step(action)
        

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
            alpha * (reward + GAMMA*np.max(q_table[t1_value_index]) -\
            q_table[update_index])
        observation = new_observation

        epsilon = MIN_EPSILON + (MAX_EPSILON-MIN_EPSILON)*np.exp(-DECAY*parameter_decay)
        alpha = MIN_ALPHA + (MAX_ALPHA-MIN_ALPHA) * np.exp(-DECAY*parameter_decay)

    recent_rewards.append(reward)
    transition_incrementer += 1

    all_Q[...,episode] = q_table 
    
	np.save(str(seed)+"_all_Q", all_Q)   
