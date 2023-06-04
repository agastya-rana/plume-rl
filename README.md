plume-rl
==============================

Solve odor plume navigation problems with RL. The odor navigation environment (the plume itself, navigator's action space and state space) are defined in src/environment.
Models used to solve this problem are in src/models. Scripts used to train these models are in src/training_scripts.

### TODO
Agastya is working on the following tasks to upgrade the codebase:
* Adding code to simulate more odor plumes
* Adding parallelization ability; figuring out how to correctly use VecEnv in SB3 to parallelize rollouts.
* Adding more odor features to the state space as required.
* Adding functionality to plot metrics of training, e.g. training loss, average rewards over time etc.

If we want to explore different action spaces from the fly (e.g. to conduct 'ablation' studies), it might
be worth defining new environments.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    ├── notebooks          <- Jupyter notebooks.
    │   └── explore_training.ipynb    <- Trains models through q learning
    ├── reports            <- Generated analysis
    │   └── figures        <- Generated graphics and figures to be used in reporting
    ├── requirements.txt   <- package requirements -- needs updating
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data
    │   │
    │   ├── models         <- Code to make plume environments
    │   │    ├── gym_motion_environment_classes.py      <- OpenAI Gym Environment for a moving odor plume
    │   │    └── odor_senses.py          <- Generates the odor features available in the state space of the agent.
    │   │    └── odor_plume.py          <- Generates the odor plume that the agent navigates in.
    │   │    └── fly_spatial_parameters.py          <- Generates the spatial state space of the fly.
    │   ├── training_scripts         <- Scripts that train agents using different algorithms.
    │   │    ├── train_rnn_baseline.py      <- Trains a LSTM model (with Actor-Critic heads) using PPO to navigate the plume
    │   │     
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based loosely on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
