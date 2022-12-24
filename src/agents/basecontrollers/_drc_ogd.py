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

"""AdaBenchmark.agents._gpc"""
from numbers import Real
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from jax import jit

from src.utils.utils import quad_loss
from src.agents.basecontrollers.agent import DRC


class DRCOGD(DRC):
    def __init__(
            self,
            d_action,
            d_state,
            start_time: int = 0,
            cost_fn: Callable[[jnp.ndarray, jnp.ndarray], Real] = None,
            H: int = 3,
            HH: int = 2,
            lr_scale: Real = 0.005,
            decay: bool = True
    ) -> None:
        """
        Description: Initialize the dynamics of the model.

        Args:
            start_time (int):
            cost_fn (Callable[[jnp.ndarray, jnp.ndarray], Real]):
            H (postive int): history of the controller
            HH (positive int): history of the system
            lr_scale (Real):
            lr_scale_decay (Real):
            decay (Real):
        """

        cost_fn = cost_fn or quad_loss

        self.t = 0  # Time Counter (for decaying learning rate)

        self.H, self.HH = H, HH

        self.lr_scale, self.decay = lr_scale, decay

        self.bias = 0

        # Model Parameters
        # initial linear policy / perturbation contributions / bias

        self.M = jnp.zeros((H, d_action, d_state))

        # Past H + HH states ordered increasing in time
        self.state_history = jnp.zeros((H + HH, d_state, 1))

        # past state and past action
        self.state, self.action = jnp.zeros((d_state, 1)), jnp.zeros((d_action, 1))

        def last_h_states():
            """Get noise history"""
            return jax.lax.dynamic_slice_in_dim(self.state_history, -H, H)

        self.last_h_states = last_h_states

        def policy_loss(M, s, G, u):
            """
            DRC loss as according to Simchovitz
            :param M:
            :param s:
            :param G:
            :param u:
            :return:
            """
            """Surrogate cost function"""

            def action(h):
                """Action function"""

                b = jnp.tensordot(
                    M, jax.lax.dynamic_slice_in_dim(s, h, H), axes=([0, 2], [0, 1])
                )
                u = jnp.clip(b, -2, 2)
                return u

            d = np.zeros(d_state)
            for i in range(H):
                #
                #   print(f'i G{self.G[i]} u {u.shape} final{(self.G[i] @ u).shape}')
                d += G[i]@action(i)  # works because G is not inverted

                #d += G[i] @M[i]@action(i) #impossible
            final_state = s[-1] + d
            return cost_fn(final_state, action(H))

        self.policy_loss = policy_loss
        self.grad = jit(grad(policy_loss, (0, 1)))

    def __call__(self, state: jnp.ndarray, G: jnp.ndarray) -> jnp.ndarray:
        """
        Description: Return the action based on current state and internal parameters.

        Args:
            state (jnp.ndarray): current state

        Returns:
           jnp.ndarray: action to take
        """

        action = self.get_action()
        self.update(G, state)
        # print(f'Action {action.shape}')
        return action

    def update(self, G: jnp.ndarray, state: jnp.ndarray, u) -> None:
        """
        Description: update agent internal state.

        Args:
            state (jnp.ndarray):

        Returns:
            None
        """
        self.state_history = self.state_history.at[0].set(state)
        self.state_history = jnp.roll(self.state_history, -1, axis=0)

        delta_M, delta_bias = self.grad(self.M, self.state_history, G, self.u)

        lr = self.lr_scale
        lr *= (1 / (self.t + 1)) if self.decay else 1
        self.M -= lr * delta_M
        self.bias -= lr * delta_bias

        # update state
        self.state = state

        self.t += 1

    def get_action(self) -> jnp.ndarray:
        """
        Description: get action from state.

        Args:
            state (jnp.ndarray):

        Returns:
            jnp.ndarray
        """
        b = jnp.tensordot(self.M, self.last_h_states(), axes=([0, 2], [0, 1]))

        # we add clipping here, so that learning doesn't get corrupted + feedback loop
        u = jnp.clip(b, -2, 2)
        self.u = u
        return u
