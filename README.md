plume-rl
==============================

Solve odor plume navigation problems with RL

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
