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

"""AdaBenchmark.agents._adaptive"""
from numbers import Real
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from src.utils.utils import lifetime


class AdaptiveDRC:
    def __init__(
            self,
            T: int,
            base_controller,
            d_action: int = 10,
            d_state: int = 10,
            cost_fn: Callable[[jnp.ndarray, jnp.ndarray], Real] = None,
            HH: int = 10,
            H: int = 10,
            eta: Real = 0.5,
            eps: Real = 1e-6,
            inf: Real = 1e6,
            life_lower_bound: int = 100,
            expert_density: int = 64,
    ) -> None:

        self.n, self.m = d_state, d_action

        self.H, self.HH = H, HH
        self.cost_fn = cost_fn
        self.base_controller = base_controller

        # Start From Uniform Distribution
        self.T = T + 1
        self.weights = np.zeros(T)
        self.weights[0] = 1.0

        # Track current timestep
        self.t, self.expert_density = 0, expert_density

        # Store Model Hyperparameters
        self.eta, self.eps, self.inf = eta, eps, inf

        # State and Action
        self.x, self.u = jnp.zeros((self.n, 1)), jnp.zeros((self.m, 1))

        # Alive set
        self.alive = jnp.zeros((T,))

        # Precompute time of death at initialization
        self.tod = np.arange(T)
        for i in range(1, T):
            self.tod[i] = i + lifetime(i, life_lower_bound)
        self.tod[0] = life_lower_bound  # lifetime not defined for 0

        # Maintain Dictionary of Active Learners
        self.learners = {}
        self.G = None
        self.state_size = d_state
        self.action_size = d_action
        self.learners[0] = base_controller(
            self.m, self.n, self.t, self.cost_fn, self.H, self.HH,
            self.eta)

        self.state_history = jnp.zeros((HH, self.n, 1))

        self.HH = HH
        self.H = H

        def policy_loss(controller, s, G):
            """Surrogate cost function"""

            def action(h):
                """Action function"""

                b = jnp.tensordot(
                    controller.M, jax.lax.dynamic_slice_in_dim(s, h, H), axes=([0, 2], [0, 1])

                )
                u = jnp.clip(b, -2, 2)
                return u

            d = np.zeros(self.n)
            for i in range(H):
                d += G[i] @ action(i)

            final_state = s[-1] + d
            return cost_fn(final_state, action(H))

        self.policy_loss = policy_loss

    def update(self, G, x):

        play_i = np.argmax(self.weights)
        self.u = self.learners[play_i].get_action()

        self.state_history = self.state_history.at[0].set(x)
        self.state_history = jnp.roll(self.state_history, -1, axis=0)

        for i in np.nonzero(self.alive)[0]:
            i = int(i)
            loss_i = self.policy_loss(self.learners[int(i)], self.state_history, G)
            self.weights[i] *= np.exp(-self.eta * loss_i)
            self.weights[i] = min(max(self.weights[i], self.eps), self.inf)
            self.learners[i].update(G, x, self.u)  # since assume those to be changing

        self.t += 1

        # One is born every expert_density steps
        if self.t % self.expert_density == 0:
            self.alive = self.alive.at[self.t].set(1)
            self.weights[self.t] = self.eps
            self.learners[self.t] = self.base_controller(
                self.m, self.n, self.t, self.cost_fn, self.H, self.HH,
                self.eta)

            self.learners[self.t].state = x
            self.learners[self.t].state_history = self.learners[self.t].state_history.at[0].set(x)

        # At most one dies
        kill_list = jnp.where(self.tod == self.t)
        if len(kill_list[0]):
            kill = int(kill_list[0][0])
            if self.alive[kill]:
                self.alive = self.alive.at[kill].set(0)
                del self.learners[kill]
                self.weights[kill] = 0

        # Rescale
        max_w = np.max(self.weights)
        if max_w < 1:
            self.weights /= max_w


        # Update System
        self.x, self.G = x, G

        return self.u
