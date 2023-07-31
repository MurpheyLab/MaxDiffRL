## Installation
- Simulations were run with Python 3.6
- Python dependencies can be run with `pip install -r requirements.txt`
- Note: specified mujoco_py version must be used for code to work. Other versions can be modified at your own discretion (e.g. PyTorch with CPU only or with a specific CUDA version)

## Running the code

To train a policy, modify the relevant yaml file for your test scenario (`config/<method>.yaml`). Then, run the following with the desired arguments (possible arguments can be viewed with `python train.py --help`)
```
python train.py --env PointMass2D_DoubleIntEnv --method maxdiff --seed 13 --beta 0.1
python train.py --env SwimmerEnv_v3 --method maxdiff --seed 13
```

To playback the learned policy, run the following with the arguments matching your training configuration (possible arguments can be viewed with `python train.py --help`)
```
python enjoy.py --env PointMass2D_DoubleIntEnv --method maxdiff --mod _H30_alpha5 --beta 0.1
python enjoy.py --env SwimmerEnv_v3 --method maxdiff --seed 13 --mod _H30
```

Results can also be visualized in the jupyter-notebooks provided in the `notebooks` folder.

## Repository Contents

```
.
├── config/                       # contains environment-specific test setup yaml files
├── data/                         # contains models and data to run notebooks
├── envs/
│   ├── __init__.py
│   ├── build_env.py              # helper function to build environments for training and testing
│   ├── light_swimmer.xml         # specifies alternate physics model (light tail)
│   ├── normalized_actions.py     # wrapper for gym environments to normalize action space
│   ├── pointmass_lib/            # folder contains functions to generate point mass environment and for rendering pointmasss figures
│   ├── replay_buffer.py          # memory buffer for all training all policies
│   └── wrappers.py               # wrappers for gym environments to modify reward functions
├── mpc_lib/                      # model predictive control policy files
│   ├── __init__.py
│   ├── entropy.py                # helper functions for MaxDiff
│   ├── max_diff.py               # MaxDiff policy module        
│   ├── model.py                  # model for MPPI and MaxDiff
│   ├── mppi.py                   # MPPI policy module
│   ├── optimizer.py              # model optimizer for MPPI and MaxDiff
│   └── utils.py                  # helper functions for max_diff and mppi
├── notebooks/                    # jupyter-notebooks for plotting reward curves
├── sac_lib/                      # model-free policy files
│   ├── __init__.py
│   ├── policynetwork.py          # policy network
│   ├── sac_orig.py               # SAC training module (with optimizer)
│   └── sac_networks.py           # SAC training networks
├── LICENSE
├── enjoy.py                      # replay of trained policies
├── README.md
├── requirements.txt              # python dependencies
├── train.py                      # main training function
└── utils.py                      # helper functions
```

## Copyright and License

The implementations of MaxDiff contained herein are copyright (C) 2023 - 2024 by Allison Pinosky and are distributed under the terms of the GNU General Public License (GPL) version 3 (or later). Please see the LICENSE for more information.

Contact: apinosky@u.northwestern.edu

Lab Info:  
Todd D. Murphey  
https://murpheylab.github.io/  
Northwestern University  
