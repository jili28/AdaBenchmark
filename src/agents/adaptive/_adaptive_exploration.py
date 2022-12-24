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
from src.systemidenfication.hokalman import systemident
from src.agents.adaptive._adaptive import Adaptive
from src.agents.basecontrollers.ada_pred import AdaPred



class AdaptiveExploration:
    def __init__(
            self,
            T: int,  # number of agents
            base_controller, environment,
            HH: int = 10, p: float = 0.1, d_state: int = 10, d_control: int = 10,
            h: int = 10, eta: Real = 0.5, eta_pred: Real = 0.001, eps: Real = 1e-6, inf: Real = 1e6,
            R_norm: Real = 0.5,  # big part that needs to be controlled
            R_nat: Real = 2.0, life_lower_bound: int = 100, expert_density: int = 64,
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

        A, B = self.get_system_matrices()
        # A = jnp.array([[1, 1], [0, 1]])
        # B = jnp.array([[0], [1]])

        self.explorer = Adaptive(T, base_controller,
                                 A, B, cost_fn, HH, h,
                                 eta, eps, inf,
                                 life_lower_bound, expert_density,
                                 Q=self.env.Q, R=self.env.R)
        # Track current timestep
        self.t = 0
        self.t_h = 0
        self.u_window = []
        self.b = 1  # first force initial system identification phase
        self.use_x_nat = use_x_nat

    def __call__(self):

        if self.b:
            self.u = np.random.choice([-1, 1], self.m)
        else:
            # play according to controller
            self.u = self.explorer.u.squeeze(1)  # actions may have different shaes
            # test whether rounding like this suffices
            # the clipping also happens in the environment anyways
            if np.abs(self.u) > 2:
                self.u = np.ones_like(self.u) * 2 * np.sign(self.u)
        # print(f'self.u played{self.u.shape}')
        played_action = self.u
        x, reward, terminated, truncated, info = self.env.__call__(self.u)

        # self.env.render()
        # print(f'X {x.shape}')
        if len(self.u_window) >= self.h:
            self.u_window.pop(0)
        self.u_window.append(self.u)
        l = np.zeros(self.n)
        for i, u in enumerate(self.u_window):
            #
            #   print(f'i G{self.G[i]} u {u.shape} final{(self.G[i] @ u).shape}')
            l += self.G[i] @ u
        # print(f'l {l.shape}')
        # print(x.shape)
        # print(f'xnat {x_nat.shape}')
        if self.use_x_nat:
            x_nat = (x - l)
        else:
            x_nat = self.round(x)
        A, B = self.get_system_matrices()

        # A = jnp.array([[1, 1], [0, 1]])
        # B = jnp.array([[0], [1]])

        self.u = self.explorer.__call__(A, B, jnp.expand_dims(x_nat, axis=1))

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
        # print(x.shape)
        for i, a in enumerate(np.abs(x)):

            if a > self.K_bound[i]:
                ret.append(self.K_bound[i] * np.sign(x))
            else:
                ret.append(a)
        return jnp.array(x)

    def get_system_matrices(self):
        G = np.flip(self.G, 0).transpose(1, 0, 2).reshape(self.n, -1)
        p = int(G.shape[1] / self.h)
        A, B = systemident(G, p)
        return A, B


def get_adaptive_exploration(args):
    agent = AdaptiveExploration(args['T'], args['base_controller'], args['env'],
                                HH=args['HH'], p=args['p'], d_state=args['env'].d_s,
                                d_control=args['env'].d_a,
                                h=args['h'], eta=args['eta'], eta_pred=args['eta_pred'],
                                eps=args['eps'], inf=args['inf'], R_norm=args['R_norm'],
                                R_nat=args['R_nat'], life_lower_bound=args['life_lower_bound'],
                                expert_density=args['expert_density'], use_x_nat=args['use_x_nat'])

    return agent
