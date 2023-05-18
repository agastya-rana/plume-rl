## This file has code to train the VAE assuming the optimal RNN is already trained.
## I need to simulate the VAE training episodes while simulating the RNN so I can capture the history dependence;
from torch.utils.data import DataLoader, TensorDataset
from ..src.vae import *
from ..src.rnn_baseline import *

def predict_proba(model, state):
    obs = obs_as_tensor(state, model.policy.device)
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().numpy()
    return probs_np

def epsilon_greedy_action(policy_probs, epsilon):
    ## Chooses an action based on epsilon-greedy strategy
    if np.random.random() < epsilon:
        # Random action
        return np.random.choice(len(policy_probs))
    else:
        # Greedy action
        return np.argmax(policy_probs)

## Change the above function to save the VAE_input, VAE_output as numpyarray, and ensure they are nsteps = 100000 long each
def get_training_data(gym_env, RNN_model, hist_len, num_actions, epsilon, epochs, nsteps=100000):
    ## Note that the hist_len is the number of timesteps including the current timestep in the history
    VAE_input = np.zeros_like((nsteps, hist_len * gym_env.observation_space.shape[0]))
    VAE_output = np.zeros_like((nsteps, gym_env.action_space.n))
    step_count = 0
    while step_count < nsteps:
        state = gym_env.reset()
        history = [state] * hist_len
        done = False
        while not done:
            # Prepare the current state and history for the RNN model
            RNN_input = np.array(history).reshape(1, -1)
            # Get the action probabilities from the RNN model
            action_probs = predict_proba(RNN_model, RNN_input)
            # Choose an action based on epsilon-greedy strategy
            action = epsilon_greedy_action(action_probs, epsilon)
            # Step the gym environment and get the new state
            new_state, reward, done, _ = gym_env.step(action)
            # Update the history
            history.append(new_state)
            history = history[-hist_len:]
            # Save the state + history (input of VAE)
            VAE_input[step_count] = np.array(history).reshape(1, -1)
            # Save the probabilities of each action (the policy; output of the VAE)
            VAE_output[step_count] = action_probs
            step_count += 1
    return VAE_input, VAE_output

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

train_vae(model, dataloader, optimizer, epochs)
