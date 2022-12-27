plume-rl
==============================

Solve odor plume navigation problems with RL.

### Notes
The jupyter notebook 'notebooks/basic_q_learning.ipynb' shows how to train a basic plume-navigating agent.
It's a good place to start to see the methods of the environment class (based on openAI Gym environments).

### To Do
The current implementation is very basic. On a technical side, it would be good to add:
* the ability to run multiple agent simulations in parallel on the same plume
* turn the training regime into a function
* add other training approaches (parameterize a continuous function from states/actions to values, for example)
* add other plume movies and simulations

On the non-technical side
* what if agent orients its sensors?
* what if agent has an orientation that is updated with a turning-based action space?
* where are gradients vs motions especially useful?
* how to perform well across diverse plume types?
* incorporate odor hit frequency and intermittency
* action and stimulus history more broadly (may require a deep RL approach if the state space grows)


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
    │   │    └── motion_environment_factory.py          <- Parameterize motion plume classes by providing a
    │   │                                                   reward structure, a state space
    │   │     
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based loosely on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
