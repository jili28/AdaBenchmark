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

"""AdaBenchmark.agents._adaptive"""
from numbers import Real
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from src.utils.utils import lifetime, quad_loss


class Adaptive:
    def __init__(
            self,
            T: int,
            base_controller,
            A: jnp.ndarray,
            B: jnp.ndarray,
            cost_fn: Callable[[jnp.ndarray, jnp.ndarray], Real] = None,
            HH: int = 10,
            H: int = 10,
            eta: Real = 0.5,
            eps: Real = 1e-6,
            inf: Real = 1e6,
            life_lower_bound: int = 100,
            expert_density: int = 64,
            Q: jnp.ndarray = None,
            R: jnp.ndarray = None,
    ) -> None:

        self.A, self.B = A, B
        self.n, self.m = B.shape

        cost_fn = cost_fn or quad_loss
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
        state_size, action_size = B.shape
        self.Q = Q
        self.R = R
        self.state_size = state_size
        self.action_size = action_size
        self.learners[0] = base_controller(A, B, cost_fn=cost_fn, Q=Q, R=R,
                                           K=jnp.ones((action_size, state_size)),
                                           H=H, HH=HH)

        self.w = jnp.zeros((HH, self.n, 1))

        self.HH = HH
        self.H = H

        def policy_loss(controller, A, B, x, w):
            def evolve(x, h):
                """Evolve function"""
                return A @ x + B @ controller.get_action(x) + w[h], None

            final_state, _ = jax.lax.scan(evolve, x, jnp.arange(HH))
            return cost_fn(final_state, controller.get_action(final_state))

        self.policy_loss = policy_loss

    def __call__(self, A, B, x):

        play_i = np.argmax(self.weights)
        self.u = self.learners[play_i].get_action(x)

        # Update alive models
        for i in np.nonzero(self.alive)[0]:
            i = int(i)
            loss_i = self.policy_loss(self.learners[int(i)], A, B, x, self.w)
            self.weights[i] *= np.exp(-self.eta * loss_i)
            self.weights[i] = min(max(self.weights[i], self.eps), self.inf)
            self.learners[i].update(x, u=self.u, A=A, B=B)  # since assume those to be changing

        self.t += 1

        # One is born every expert_density steps
        if self.t % self.expert_density == 0:
            self.alive = self.alive.at[self.t].set(1)
            self.weights[self.t] = self.eps
            # self.learners[self.t] = self.base_controller(A, B, cost_fn=self.cost_fn, K=self.on)
            # self.learners[self.t]  = self.base_controller(A, B, cost_fn=self.cost_fn,
            #                                               Q=self.Q, R=self.R,
            #                                               K=jnp.ones((self.action_size, self.state_size)))

            self.learners[self.t] = self.base_controller(A, B, cost_fn=self.cost_fn,
                                                         Q=self.Q, R=self.R,
                                                         H=self.H, HH=self.HH)

            self.learners[self.t].x = x

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

        # Get new noise (will be located at w[-1])
        self.w = self.w.at[0].set(x - self.A @ self.x + self.B @ self.u)
        self.w = jnp.roll(self.w, -1, axis=0)  # sets new noise

        # Update System
        self.x, self.A, self.B = x, A, B

        return self.u
