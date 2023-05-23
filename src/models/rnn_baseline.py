from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

def define_model(env, feature_dim: int = 256):
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, policy_kwargs={"lstm_hidden_size":feature_dim})
    return model
