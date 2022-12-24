# Copyright 2022 Jieming Li.
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


# To Do recheck order of G
"""AdaBenchmark.agents._adaptive"""
from numbers import Real

import jax.numpy as jnp
import numpy as np

from ._adaptive_drc import AdaptiveDRC
from src.agents.basecontrollers.ada_pred import AdaPred


class AdaptiveExplorationPartial:
    def __init__(
            self,
            T: int,  # number of agents
            base_controller, environment,
            HH: int = 10, p: float = 0.1, d_state: int = 10, d_control: int = 10,
            h: int = 10, eta: Real = 0.5, eta_pred: Real = 0.001, eps: Real = 1e-6, inf: Real = 1e6,
            R_norm: Real = 0.5,
            life_lower_bound: int = 100,
            expert_density: int = 64,
            use_x_nat=False
    ) -> None:

        self.n, self.m = d_state, d_control
        self.p = p

        self.h = h
        cost_fn = environment.get_cost_function()

        self.env = environment

        # Initialize AdaPred
        self.K_bound = self.env.bounds()
        self.pred_controller = AdaPred(h, d_state, d_control, R_norm,
                                       p=0.1,
                                       eta=eta_pred)
        self.G = self.pred_controller.get_estimate()

        self.expert_density = expert_density
        self.life_lower_bound = life_lower_bound
        self.explorer = AdaptiveDRC(T, base_controller, d_control, d_state, cost_fn=cost_fn,
                                    H=h, HH=HH, eta=eta, life_lower_bound=life_lower_bound,
                                    expert_density=expert_density)

        self.t = 0
        self.t_h = 0
        self.u_window = []
        self.b = 1  # first force initial system identification phase
        self.use_x_nat = use_x_nat

    def __call__(self):

        if self.b:
            self.u = np.random.choice([-1, 1], self.m)
        else:
            self.u = self.explorer.u.squeeze(axis=1)
            if np.abs(self.u) > 2:
                self.u = np.ones_like(self.u) * 2 * np.sign(self.u)

        played_action = self.u
        x, reward, terminated, truncated, info = self.env.__call__(self.u)

        if len(self.u_window) >= self.h:
            self.u_window.pop()
        self.u_window.append(self.u)
        x_w = np.zeros(self.n)

        for i, u in enumerate(self.u_window):
            x_w += self.G[i] @ u

        if self.use_x_nat:
            x_nat = (x - x_w)
        else:
            x_nat = self.round(x)

        if x_nat.ndim < 2:
            x_nat = jnp.expand_dims(x_nat, axis=1)

        # self.explorer.update(self.G, x_nat)
        self.explorer.update(self.G, x_nat)

        if self.t % self.h == 0 and self.t > 0:
            if self.b:
                # feed into estimator

                G = [jnp.matmul(jnp.expand_dims(x, axis=1),
                                jnp.expand_dims(u, axis=1).transpose(1, 0))
                     for u in self.u_window]  # has to be reversed as G[0] is assumed to be newest
                # G = jnp.array(G.reverse())  # consider whether order is correct
                G = jnp.array(G)
                self.pred_controller(self.b, G)
            else:
                # adds pruning etc
                self.pred_controller(self.b, None)

            self.b = np.random.binomial(1, self.p)
            self.G = self.pred_controller.get_estimate()
        self.t += 1
        return self.u, played_action, x, reward, terminated, truncated

    def round(self, x):
        ret = []
        for i, a in enumerate(np.abs(x)):

            if a > self.K_bound[i]:
                ret.append(self.K_bound[i] * np.sign(x))
            else:
                ret.append(a)
        return jnp.array(x)


def get_adaptive_exploration_partial(args):
    agent = AdaptiveExplorationPartial(args['T'], args['base_controller'], args['env'],
                                       HH=args['HH'], p=args['p'], d_state=args['env'].d_s,
                                       d_control=args['env'].d_a,
                                       h=args['h'], eta=args['eta'], eta_pred=args['eta_pred'],
                                       eps=args['eps'], inf=args['inf'], R_norm=args['R_norm'],
                                       life_lower_bound=args['life_lower_bound'],
                                       expert_density=args['expert_density'],
                                       use_x_nat=args['use_x_nat'])

    return agent


def get_adaptive_exploration_partial_sweep(sweep_dict, args):
    agent = AdaptiveExplorationPartial(args['T'], args['base_controller'], args['env'],
                                       HH=sweep_dict.HH, p=sweep_dict.p, d_state=args['env'].d_s,
                                       d_control=args['env'].d_a,
                                       h=sweep_dict.h, eta=sweep_dict.eta, eta_pred=sweep_dict.eta_pred,
                                       eps=sweep_dict.eps, inf=sweep_dict.inf, R_norm=sweep_dict.R_norm,
                                       life_lower_bound=sweep_dict.life_lower_bound,
                                       expert_density=sweep_dict.expert_density,
                                       use_x_nat=sweep_dict.use_x_nat)
    return agent
