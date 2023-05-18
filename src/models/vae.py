import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
## Goal here is to construct a VAE that takes as input a set of odor statistics, and a history,
## We use a TCN to encode temporal embeddings; then, we use a VAE on the output space to create our latent space.
## TCN architecture explained in https://arxiv.org/pdf/1803.01271.pdf


num_stats = 3 ## Concentration, motion, gradient
history_dep = 0 ## number of timesteps of history incorporated

input_dim = num_stats*(history_dep + 1)
filter_len = 5
hidden_layers = 3
max_dilation = 2**hidden_layers
padding = (filter_len - 1) * max_dilation

## In running this, need to supply inputs in the format (features, time); this will be transposed in the code

## Deletes extra padding on the right-side (future padding is not needed, since TCNs are causal)
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

## A single TCN residual block consisting of two dilation networks
class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels) ## Note that num_channels is the size of the output layer at each step.
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # Transpose to (batch_size, n_features, sequence_length)
        x = x.transpose(1, 2)
        y = self.network(x).transpose(1,2)
        return y

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.fc_layers.extend([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


## Need to change my decoder and loss function to output the policy; train it on the optimal policy chosen by the
## RNN actor-critic structure

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList([nn.Linear(latent_size, hidden_size)])
        self.fc_layers.extend([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        x = torch.sigmoid(self.fc_out(x))
        return x


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers_encoder, num_layers_decoder):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size, num_layers_encoder)
        self.decoder = Decoder(latent_size, hidden_size, input_size, num_layers_decoder)

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

# Compute loss and perform backpropagation for training
# ... (code for loss calculation and backpropagation)
