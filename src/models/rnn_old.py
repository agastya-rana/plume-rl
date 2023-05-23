## Here, we train an RNN to compute the optimal policy using Actor-Critic architecture along with PPO to train, analogous to Brunton paper
## We use stable_baselines3 here instead of the from-scratch code in https://arxiv.org/pdf/2109.12434.pdf

import torch
from torch import nn
import gym
from typing import List, Optional, Tuple, Type, Union, Dict, Callable, Any
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy, register_policy, ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class RNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 256):
        super(RNNFeaturesExtractor, self).__init__(observation_space, features_dim)

        self.hidden_dim = features_dim
        self.rnn = nn.GRU(
            input_size=observation_space.shape[0],
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        ## Might be harder to implement this actually since what we want is for RNN to take as input the history of the episode and to reset at the end of each episode.
        ## Problem with RNN as implemented here is that it only takes last timestep, since sequence length here is implicitly 1.
        ## Note that the dimension mismatch is probably occuring only because the feature extractor is not defined within the policy but rather overriden later; earlier in init, feature extractor was given input dimensions via flatten of 7

        # We init with zeros because we want the features_extractor to work for any episode length
        # We detach the hidden state to avoid backpropagating through the entire episode history
        # NOTE: the batch size is infered from the observations size
        batch_size = observations.shape[0]
        h_0 = torch.zeros((1, batch_size, self.hidden_dim))
        _, h_n = self.rnn(observations, h_0.detach())
        return h_n.squeeze(0)
    
        batch_size = observations.shape[0]
        hidden_state = torch.zeros(1, batch_size, self.hidden_dim)
        out, _ = self.rnn(observations.unsqueeze(1), hidden_state)
        return out.squeeze(1)

class RNNActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self, 
        observation_space: gym.spaces.Space, 
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        features_dim: int = 256,  # add a parameter for features_dim
        *args, 
        **kwargs
    ):
        super(RNNActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs
        )

        # Override the features extractor with our custom RNN
        self.features_extractor = RNNFeaturesExtractor(observation_space, features_dim)


def define_model(env, feature_dim: int = 256, actor_layers=[128, 128], critic_layers=[128, 128]):
    register_policy("RNNActorCriticPolicy", RNNActorCriticPolicy)
    model = PPO("RNNActorCriticPolicy", env, verbose=1, policy_kwargs={"net_arch": [dict(pi=[feature_dim]+actor_layers, vf=[feature_dim]+critic_layers)], "features_dim":feature_dim})
    return model

def policy_probs(model, state):
    obs = obs_as_tensor(state, model.policy.device)
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().numpy()
    return probs_np