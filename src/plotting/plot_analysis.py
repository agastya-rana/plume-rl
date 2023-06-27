import os
import numpy as np
import matplotlib.pyplot as plt


## Function to load data from path
def load_data(path):
    ## Load eval data and remove NaNs
    rewards = np.load(path + "_reward_history.npy")
    success = np.load(path + "_success_history.npy")
    states = np.load(path + "_state_history.npy")
    actions = np.load(path + "_action_history.npy")
    print("Success Rate:", np.mean(success))
    ## Select only the states and actions that are not NaN
    nan_idx = np.isnan(states[:,:,0])
    states = states[~nan_idx]
    actions = actions[~nan_idx]
    return rewards, success, states, actions

## Function to plot the reward history
def plot_smoothed_reward_history(rewards, window=100):
    ## Calculate the moving average
    moving_avg = np.convolve(rewards, np.ones(window), 'valid') / window
    ## Plot the moving average
    plt.plot(moving_avg)
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.show()

## Function to plot the state pdfs
def plot_state_pdfs(states, state_names, num_bins=100, exclude_zero=False):
    sname = ['conc', 'grad', 'hrc']
    fig, axes = plt.subplots(1, 3, figsize = (10,3))
    if exclude_zero:
        ## Set zero values to NaN
        states[states==0] = np.nan
    for i, ax in enumerate(axes.flatten()):
        ## Exclude Nan
        state = states[:,i]
        state = state[~np.isnan(state)]
        ax.hist(state, bins = 100)
        ax.set_yscale('log')
        ax.set_title(sname[i])
    plt.tight_layout()
    plt.show()

## Function to plot state pdfs conditioned on action
def plot_state_pdfs_disc_action(states, actions, state_names, num_bins=100):
    n_obs = 3
    n_action = 4
    fig, axes = plt.subplots(n_obs, n_action, figsize = (10,10))
    for obs in range(n_obs):
        for act in range(n_action):
            axes[obs, act].hist(states[actions==act,obs], bins = 100)
            axes[obs, act].set_title(state_names[obs]+' | Action '+str(act))
            axes[obs, act].set_yscale('log')
    plt.tight_layout()