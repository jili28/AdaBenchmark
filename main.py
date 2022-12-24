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

"""PPO."""
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
from src.utils.model_factory import get_model
from config import config

"""
Factors to consider
does K-stability hold nicely in pendulum?
Consider R-norm
maybe smaller windows?

can we use the other losses in the classical control section (look at PPG etc)

--> sanity check for different gravities, ask for esimation balacne etc
recheck for bugs, learner? balances
--> look whether output bounding exists
I also think that it should be provable, that linear 
dynamics might not suffice to control for cases like the 
pendulum if we can't do fast strong improvements

sampling vs to adaptively guarantee changing system matrics
"""

T = 500  # number of rounds
HH = 10  #
p = 0.15
h = 3
eta = 0.05
eta_pred = 0.005
eps = 1e-5
inf = 1e6
R_norm = 5  # 2 most interesting until now #4
R_nat = 3.14
life_lower_bound = 100
expert_density = 16


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
    # agent  = get_adaptive_exploration(config_dict)
    # agent = AdaptiveExploration(T, GPC, env,
    #                             HH, p, d_state, d_control,
    #                             h, eta, eta_pred, eps, inf, R_norm, R_nat, life_lower_bound, expert_density)

    trajectories = []
    for episode in range(config['episodes']):

        logging.info("Start new Episode")
        config_dict['env'] = config['env'](g=config['gravity'], states=config['env_mode'])  # create environment
        config_dict['T'] = config['T']
        config_dict['gravity'] = config['gravity']
        agent = get_model(modelname, config_dict)

        # set up new logging
        run_dir = log_dir + f'/run{episode}'
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        writer = SummaryWriter(log_dir=run_dir)
        w = wandb.init(project=f"AdaExploration", reinit=True,
                       config=config_dict,
                       id=f"{run_id}_{modelname}_{episode}")

        wandb.define_metric("Reward", summary="mean")
        total_reward = []
        window = []
        for epoch in range(config['T']):

            next_action, played_action, x, reward, terminated, truncated = agent.__call__()

            writer.add_scalar("Reward", reward, epoch)
            writer.add_scalar("Control", played_action, epoch)
            writer.add_scalar("Angle", x[0], epoch)
            writer.add_scalar("Velocity", x[1], epoch)

            if next_action > 100:
                wandb.log({"Epoch": epoch, "failure": 1})
                print(f"Episode {episode} had feedback loop: {next_action}")
                break

            window.append(reward)
            if len(window) > 30:
                window.pop(0)
            wandb.log({"Epoch": epoch, "Reward": reward, 'Control': played_action,
                       "Next_Control": next_action, "Angle": x[0], "Velocity": x[1],
                       "RewardRunningMean": np.mean(window)})

            total_reward.append(reward)

            if terminated or truncated:
                print("Truncated/Terminated")
                break
        w.summary["Mean Reward"] = np.mean(total_reward)
        w.finish()
        trajectories.append(total_reward)

    a = np.array(trajectories)
    np.savez(log_dir + "/trajectories.npz", a)
    std = np.std(a, axis=0)
    mean = np.mean(a, axis=0)
    print(f'Mean Reward{np.mean(a)}')
    print(f'Mean Std {np.std(a)}')
    x = np.linspace(0, len(std), len(std))
    y1 = mean + std
    y2 = mean - std
    np.savez(log_dir + "/array.npz", mean, std)
    fig, ax = plt.subplots()
    ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
    ax.plot(x, (y1 + y2) / 2, linewidth=2)
    if modelname == 'AdaExpGPC':
        ax.set_title(f"Mean Trajectories {modelname}")
    elif modelname == 'AdaDRC_OGD':
        ax.set_title(f"Mean Trajectories {modelname}")
    ax.set_xlabel("t")
    ax.set_ylabel("Reward")
    plt.savefig(log_dir + f"/{modelname}plt.png")


if __name__ == '__main__':
    train()
