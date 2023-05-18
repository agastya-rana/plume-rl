## Here, we train an RNN to compute the optimal policy using Actor-Critic architecture along with PPO to train, analogous to Brunton paper
## We use stable_baselines3 here instead of the from-scratch code in https://arxiv.org/pdf/2109.12434.pdf

import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy, register_policy
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
        batch_size = observations.shape[0]
        hidden_state = torch.zeros(1, batch_size, self.hidden_dim)
        out, _ = self.rnn(observations.unsqueeze(1), hidden_state)
        return out.squeeze(1)

class CustomActorCriticPolicy(BasePolicy):
    def __init__(
        self, 
        observation_space: gym.spaces.Space, 
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args, 
        **kwargs
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs
        )

        # Override the features extractor with our custom RNN
        self.features_extractor = RNNFeaturesExtractor(observation_space)


def define_model(env, actor_layers=[128, 128, 128], critic_layers=[128, 128, 128]):
    register_policy("CustomActorCriticPolicy", CustomActorCriticPolicy)
    model = PPO("CustomActorCriticPolicy", env, verbose=1, policy_kwargs={"net_arch": [dict(pi=actor_layers, vf=critic_layers)]})
    return model

def policy_probs(model, state):
    obs = obs_as_tensor(state, model.policy.device)
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().numpy()
    return probs_np