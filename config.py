from src.env.pendulum import Pendulum


config = {}


config['T'] = 1000
config['episodes'] = 10
#config['model_name'] = 'AdaExpGPC'
config['model_name'] = 'AdaDRC_OGD'
config['env'] = Pendulum
config['env_mode'] = 'full' #'angular'
config['gravity'] = 0

config['mean_window_len'] = 30