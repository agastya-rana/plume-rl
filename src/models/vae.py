## This file has code to train the VAE assuming the optimal RNN is already trained.
## I need to simulate the VAE training episodes while simulating the RNN so I can capture the history dependence;
from torch.utils.data import DataLoader, TensorDataset
from src.models.vae_arch import *
import numpy as np
import torch
import torch as th
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecEnv

def get_training_data(gym_env, model, hist_len, n_steps=100000):
    ## Environment must be vector environment
    ## Init: make history vector that is fed as inputs
    ## Make output vector that is action probabilities
    ## Over x episodes,
    ## 1. Reset environment,
    ## 2. Get state and update history,
    ## 3. Feed state to RNN and get action probabilities,
    ## 4. Choose action deterministically (because RNNs were trained on deterministic action sequencing - we can play around with this),
    ## 5. Save history and action probabilities as training data
    ## 6. Step environment and loop
    ## Note that the hist_len is the number of timesteps including the current timestep in the history
    ## Ensure environment is a vector environment
    assert isinstance(gym_env, VecEnv)
    n_envs = gym_env.num_envs
    training_input = np.zeros_like((n_steps, n_envs, (hist_len + 1) * gym_env.observation_space.shape[0]))
    training_output = np.zeros_like((n_steps, n_envs, gym_env.action_space.n))
    step_count = 0
    states = gym_env.reset()
    lstm_states = th.zeros((1, n_envs, model.policy.lstm_actor.hidden_size), device=model.policy.device))
    #env = SubprocVecEnv([make_env(i, config) for i in range(training_dict["N_ENVS"])])
    history = np.zeros((n_envs, hist_len + 1, gym_env.observation_space.shape[0]))
    while step_count < n_steps:
        if not np.all(episode_starts == False):
            for i, start in enumerate(episode_starts):
                if start:
                    history[i] = np.zeros((hist_len + 1, gym_env.observation_space.shape[0]))
                    history[i, 0] = states[i]
        distribution, lstm_states = model.policy.get_distribution(th.from_numpy(states), lstm_states, th.from_numpy(episode_starts))
        action_probs = distribution.distribution.probs
        states, _, episode_starts, _ = gym_env.step(np.argmax(action_probs))
        training_input[step_count] = history.reshape((n_envs, -1))
        training_output[step_count] = action_probs
        np.roll(history, 1, axis=1)
        history[:, 0, :] = states
        step_count += 1
    training_input = training_input.reshape((n_steps * n_envs, -1))
    training_output = training_output.reshape((n_steps * n_envs, -1))
    return training_input, training_output

def make_dataloader(VAE_input, VAE_output, batch_size):
    ## Convert them to PyTorch tensors
    VAE_input_tensor = torch.from_numpy(VAE_input).float()
    VAE_output_tensor = torch.from_numpy(VAE_output).float()
    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(VAE_input_tensor, VAE_output_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_vae(model, dataloader, optimizer, epochs):
    """
    Train a VAE model.
    model: The VAE model.
    dataloader: A PyTorch DataLoader supplying the training data.
    optimizer: The optimizer to use for training.
    epochs: The number of epochs to train for.
    """
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            # Move the data to the device (CPU or GPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data, target = data.to(device), target.to(device)
            # Reset the gradients
            optimizer.zero_grad()
            # Forward pass (encode, decode and generate sample)
            reconstructed_batch, mu, logvar = model(data)
            # Calculate loss
            loss = VAE_loss(reconstructed_batch, data, mu, logvar)
            # Backward pass
            loss.backward()
            # Perform a single optimization step
            optimizer.step()
        print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}')
