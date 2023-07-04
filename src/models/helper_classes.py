from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class FilterExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_heads=3, n_odor=1, history_len=10, theta_dim=2):
        self.n_odor = n_odor
        self.n_heads = n_heads
        features_dim = n_heads*n_odor + theta_dim ## The features dimension is the number of neurons in the first layer + the number of theta values
        self.theta_dim = theta_dim
        self.history_len = history_len + 1 ## + 1 because the current timestep is also included for easier manipulation in the forward method
        super().__init__(observation_space, features_dim)
        # Define the first layer with n_heads*n_odor neurons that is a linear filter with offset
        self.filter = nn.ModuleList([nn.Linear(self.history_len, 1) for _ in range(n_heads*n_odor)])

    def forward(self, observations):
        ## Remove the theta values from the observations
        x = observations[:, :-self.theta_dim]
        x = x.view(-1, self.history_len, self.n_odor) ## Reshape to (batch_size, history_len, n_odor)
        x = [fc(x[:, :, i // self.n_heads]) for i, fc in enumerate(self.filter)] ## List of tensors of shape (batch_size, 1)
        x = torch.cat(x, dim=1) ## Concatenate along the last dimension to get a tensor of shape (batch_size, n_heads*n_odor)
        ## Add back the theta values to x and return
        x = torch.cat((x, observations[:, -self.theta_dim:]), dim=1) ## Concatenate along the last dimension to get a tensor of shape (batch_size, n_heads*n_odor + theta_dim)
        return x

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