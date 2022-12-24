# Copyright 2022 The Deluca Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import time

import numpy as np
from matplotlib import pyplot as plt

from src.utils.logging import Logger
import wandb
from tensorboardX import SummaryWriter

from src.agents.hyperparameters import params
from src.agents.sweep_configuration import sweep_configuration
from src.utils.model_factory import get_sweep_model
from config import config

def train(workdir=None):
    """Main training loop.

  config
    - num_episodes
    - episodes_per_eval
  """
    # optim = optax.adam(learning_rate=config.learning_rate)
    # optim_state = optim.init(agent)
    run_id = time.strftime("%Y%m%d-%H%M%S")

    modelname = config['model_name']
    log_dir = f"reports/logs/{run_id}_torch_{modelname}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(print_fp=os.path.join(log_dir, "out.txt"))
    # Create logging file
    logging.basicConfig(
        filename=f"{log_dir}/info.log", level=logging.INFO
    )

    config_dict = params[modelname]

    logging.info("Started logging.")

    config_dict['env'] = config['env']()  # create environment
    config_dict['T'] = config['T']

    run = wandb.init(project=f"AdaExplorationSweep")

    wandb.define_metric("Reward", summary="mean")
    logging.info("Start new Episode")
    config_dict['env'] = config['env'](g=config['gravity'])  # create environment
    config_dict['T'] = config['T']
    config_dict['gravity'] = config['gravity']
    agent = get_sweep_model(modelname, config_dict, wandb.config)

        # set up new logging
    run_dir = log_dir
    writer = SummaryWriter(log_dir=run_dir)

    total_reward = []
    window = []
    for epoch in range(config['T']):

        next_action, played_action, x, reward, terminated, truncated = agent.__call__()

        writer.add_scalar("Reward", reward, epoch)
        writer.add_scalar("Control", played_action, epoch)
        writer.add_scalar("Angle", x[0], epoch)
        writer.add_scalar("Velocity", x[1], epoch)

        if next_action > 100:
            print(f"had feedback loop: {next_action}")
            break

        window.append(reward)
        if len(window) > config['mean_window_len']:
            window.pop(0)
        wandb.log({"Epoch": epoch, "Reward": reward, 'Control': played_action,
                   "Next_Control": next_action, "Angle": x[0], "Velocity": x[1],
                   "RewardRunningMean": np.mean(window)})

        total_reward.append(reward)

        if terminated or truncated:
            print("Truncated/Terminated")
            break

    a = np.array(total_reward)
    np.savez(log_dir + "/trajectories.npz", a)
    print(f'Mean Reward{np.mean(a)}')
    print(f'Mean Std {np.std(a)}')
    x = np.linspace(0, len(a), len(a))
    np.savez(log_dir + "/array.npz", a)
    fig, ax = plt.subplots()
    # ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
    ax.plot(x, a, linewidth=2)
    ax.set_title(f"Mean Trajectories {modelname}")
    ax.set_xlabel("t")
    ax.set_ylabel("Reward")
    plt.savefig(log_dir + f"/{modelname}plt.png")


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='AdaExplorationSwee')
    wandb.agent(sweep_id, function=train, count=40)