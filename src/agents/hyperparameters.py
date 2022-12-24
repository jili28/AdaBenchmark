
from src.agents.basecontrollers._gpc import GPC
from src.agents.basecontrollers._drc_ogd import DRCOGD

params = {
    "AdaExpGPC" :{
        # number of rounds
        'base_controller': GPC,
        'HH' : 2, #
        'p' : 0.25,
        'h' : 2,
        'eta' : 0.05,
        'eta_pred' : 0.01,
        'eps' : 1e-5,
        'inf' : 1e6,
        'R_norm' :  10, #4 ,#5, #2 most interesting until now #4
        'R_nat' : 3.14,
        'life_lower_bound' : 100,
        'expert_density' : 64,
        'use_x_nat': True #using error corrected x_nat, or exact
    },

    'AdaDRC_OGD':{
        'base_controller': DRCOGD,
        'HH' : 5, #
        'p' : 0.15,
        'h' : 3,
        'eta' : 0.01,
        'eta_pred' : 0.01,
        'eps' : 1e-5,
        'inf' : 1e6,
        'life_lower_bound' : 100,
        'expert_density' : 32,
        'R_norm' :  6, #5, #4 ,#5, #2 most interesting until now #4
        'use_x_nat': True #using error corrected x_nat, or exact
    },

    "Best" :{
        # number of rounds
        'base_controller': GPC,
        'HH' : 2, #
        'p' : 0.10,
        'h' : 2,
        'eta' : 0.1,
        'eta_pred' : 0.01,
        'eps' : 1e-5,
        'inf' : 1e6,
        'R_norm' :  5, #4 ,#5, #2 most interesting until now #4
        'R_nat' : 3.14,
        'life_lower_bound' : 100,
        'expert_density' : 64,
        'use_x_nat': True #using error corrected x_nat, or exact
    }
}
