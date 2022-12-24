from src.agents.adaptive._adaptive_exploration import get_adaptive_exploration
from src.agents.adaptive._adaptive_exploration_partial import get_adaptive_exploration_partial, get_adaptive_exploration_partial_sweep


def get_model(model_name, args):
    if model_name == 'AdaExpGPC':
        return get_adaptive_exploration(args)
    elif model_name == 'AdaDRC_OGD':
        return get_adaptive_exploration_partial(args)
    else:
        raise Exception("Model not defined")


def get_sweep_model(modelname, args, sweep_dict):
    if modelname == 'AdaDRC_OGD':
        return get_adaptive_exploration_partial_sweep(sweep_dict, args)
