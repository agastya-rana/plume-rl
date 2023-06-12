import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from rnn_baseline import *
from vae_helper_classes import *

## Goal here is to construct a VAE that takes as input a set of odor statistics, and a history,
## We use a TCN to encode temporal embeddings; then, we use a VAE on the output space to create our latent space.
## TCN architecture explained in https://arxiv.org/pdf/1803.01271.pdf

## Instead of using TCNs and learning the latent space, we can simply take each odor measurement (for example a 3 element vector at each timestep)
## and send it to a multi-head set of perceptrons that are all each only connected to one type of odor measurement
## Then, we can treat these perceptrons as the latent space, learn the 'filters' (weights of each perceptron), and then see how they work

num_stats = 3 ## Concentration, motion, gradient
history_dep = 0 ## number of timesteps of history incorporated
input_dim = num_stats*(history_dep + 1)
filter_len = 5
hidden_layers = 3
max_dilation = 2**hidden_layers
padding = (filter_len - 1) * max_dilation

## TODO: In running this, need to supply inputs in the format (features, time); this will be transposed in the code

class LinearFilter_Encoder(nn.Module):
    ## This class will have the first layer send each input timeseries to a separate neuron; this could be multiheaded?
    ## We send these multi-headed filters to the through a hidden layer, and then to the latent space 
    def __init__(self, hidden_size, latent_dim, n_features=3):
        ## Input should be a tensor of size (n_features, n_timesteps)
        super().__init__()
        self.n_heads = n_heads
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        # Define the first layer with n_heads*n_features neurons that is a linear filter with offset
        self.fc1 = nn.ModuleList([nn.Linear(n_timesteps, 1) for _ in range(n_heads*n_features)])
        # Define the second layer as a fully connected feedforward network
        self.fc2 = nn.Linear(n_heads*n_features, hidden_size)
        # Define linear layers to output mu and logvar
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        x = x.view(self.n_features, -1)
        x = [F.relu(fc(x[i%self.n_heads])) for i, fc in enumerate(self.fc1)]
        x = torch.cat(x, dim=0)
        # Apply the second layer
        x = F.relu(self.fc2(x))
        # Compute mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class TCN_MLP_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers, n_TCN_layer=5, kernel_size=5, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList([TCN(input_size, hidden_size, [hidden_size]*n_TCN_layer, kernel_size=kernel_size, dropout=dropout)])
        self.fc_layers.extend([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1 - n_TCN_layer)])
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

## Train it on the optimal policy chosen by the RNN actor-critic structure

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList([nn.Linear(latent_size, hidden_size)])
        self.fc_layers.extend([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)])
        ## Convert to action space
        self.fc_layers.append(nn.Linear(hidden_size, output_size))
        self.fc_out = nn.Softmax()
    def forward(self, x):
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        x = self.fc_out(x)
        return x


class VAE(nn.Module):
    def __init__(self, encoder_kwargs, decoder_kwargs):
        super(VAE, self).__init__()
        self.encoder = None
        self.decoder = None
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

class Filter_MLP_VAE(VAE):
    def __init__(self, encoder_kwargs, decoder_kwargs):
        super().__init__(encoder_kwargs, decoder_kwargs)
        self.encoder = LinearFilter_Encoder(**encoder_kwargs)
        self.decoder = Decoder(**decoder_kwargs)

def VAE_loss_function(reconstructed, x, mu, logvar):
    # Reconstruction loss (KL-divergence between expected action probs and optimal RNN policy)
    reconstruction_loss = nn.KLDivLoss(reconstructed, policy_probs())
    # Kullback-Leibler (KL) divergence loss
    kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Total loss
    loss = reconstruction_loss + kl_divergence_loss
    return loss

# Example usage
input_size = 32
hidden_size = 64
latent_size = 10

# Instantiate the VAE model
vae = VAE(input_size, hidden_size, latent_size)

# Generate some random input data
input_data = torch.randn(16, input_size)  # Batch size of 16

# Forward pass through the VAE
reconstructed_data, mu, logvar = vae(input_data)

# Compute the loss
loss = loss_function(reconstructed_data, input_data, mu, logvar)

# Perform backpropagation
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
optimizer.zero_grad()
loss.backward()
optimizer.step()



# Example usage
input_size = 32
hidden_size = 64
latent_size = 10
num_layers_encoder = 2
num_layers_decoder = 2
# Instantiate the VAE model
vae = VAE(input_size, hidden_size, latent_size, num_layers_encoder, num_layers_decoder)
# Generate some random input data
input_data = torch.randn(16, input_size)  # Batch size of 16
# Example usage
input_size = 32
hidden_size = 64
latent_size = 5
# Instantiate the VAE model
vae = VAE(input_size, hidden_size, latent_size)
# Generate some random input data
input_data = torch.randn(16, input_size)  # Batch size of 16
# Forward pass through the VAE
reconstructed_data, mu, logvar = vae(input_data)

