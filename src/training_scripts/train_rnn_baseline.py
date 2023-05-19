## Trains the baseline RNN on the odor plume using PPO on actor-critic MLP heads stemming from the RNN feature extractor
from ..src.rnn_baseline import *
from ..src.gym_motion_environment import *

register_policy("CustomActorCriticPolicy", CustomActorCriticPolicy)
## Define the environment here
environment = PlumeMotionNavigationEnvironmentMovie1PlumeSourceRewardStopActionFactory(config=config_dict, actions=WalkStopActionEnum, rng = rng).plume_environment
## Define the model to be run
model = define_model(environment)
# Train the model
model.learn(total_timesteps=10000)
# Save the model
model.save("rnn_baseline")
# Load the model
model = PPO.load("rnn_baseline")
# Evaluate the model
obs = env.reset()
## Check to see how done is evaluated and copy training metrics from other places
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
env.close()
