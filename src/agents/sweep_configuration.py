"""
Here we can store the hyperparameter configurations for each model
"""
from src.agents.basecontrollers._gpc import GPC
from src.agents.basecontrollers._drc_ogd import DRCOGD

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'Reward'},
    'parameters': {
        'HH': {'values': [1, 2, 4, 8]},
        'p': {'values': [0.15, 0.2, 0.25, 0.3]},
        'h': {'values': [1, 2, 4]},
        'eta': {'values': [0.001, 0.005, 0.01]},
        'eta_pred': {'values': [0.001, 0.005, 0.01]},
        'eps': {'values': [1e-5]},
        'inf': {'values': [1e6]},
        'life_lower_bound': {'values': [100, 200, 300]},
        'expert_density': {'values': [32, 64, 128]},
        'R_norm': {'values': [4, 8, 16]},  # 5, #4 ,#5, #2 most interesting until now #4
        'use_x_nat': {'values': [True, False]}  # using error corrected x_nat, or exact
    }
}
