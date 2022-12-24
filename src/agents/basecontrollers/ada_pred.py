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

from numbers import Real
import jax.numpy as jnp
import numpy as np
import math



def getRightSetBit(n):
    return int(math.log2(n & -n) + 1)


class AdaPred:
    def __init__(
            self,
            h,
            n_state,
            n_control,
            R, #operator bound
            p: Real=0.1,
            eta: Real=0.1
    ) -> None:

        # working dictionary
        self.K = (h, n_state, n_control)

        self.h = h
        self.R = R
        self.s = {1: np.random.random(self.K)}  # decision set
        self.p = p
        self.eta = eta
        self.reset()

    def reset(self):
        self.t = 1
        self.s = {1: self.sample()}
        self.q = {1: 1}

    def sample(self):
        matrices = np.random.random((self.K))
        return self.project(matrices)

    def project(self, m):
        #print(m.shape)
        norms = jnp.linalg.norm(m, ord=2, axis=(1, 2))
        s = jnp.linalg.norm(norms, ord=1)
        if self.R < s:
            m = m * self.R / s
        return m

    def get_estimate(self):
        m = jnp.zeros(self.K)
        #print(f'm {m.shape}')
        for keys in self.q.keys():
            m += self.q[keys] * self.s[keys]
        return m

    def __call__(self, b, g):
        if b == 1:
            def loss(z, g):
                # print(f'{(g - z).shape}')
                return 1 / (2 * self.p) * np.linalg.norm(g- z)  # returns ravel as expected

            def loss_grad(z, g):
                return 1 / self.p * (z - g)
        else:
            def loss(z, g):
                return 0

            def loss_grad(z, g):
                return 0

        new_s = {}
        new_q = {}
        norm = 0
        for i in self.s.keys():  # we ignore projection for now
            new_s[i] = self.project(self.s[i] - self.eta * loss_grad(self.s[i], g))
            new_q[i] = self.q[i] * np.exp(-self.eta * loss(self.s[i], g))
            norm += new_q[i]

        for i in new_q.keys():
            new_q[i] = new_q[i] * (self.t / self.t + 1) / norm

        # add new instance
        new_s[self.t + 1] = self.sample()
        new_q[self.t + 1] = 1 / (self.t + 1)

        # pruning
        keys = new_s.keys()
        for i in keys:
            k = getRightSetBit(i)
            m = math.pow(2, k + 2) + 1
            if self.t > i + m:
                new_s.pop(i)
            else:
                pass
        # renormalize
        norm = 0
        for i in new_q.keys():
            norm += new_q[i]
        for i in new_q.keys():
            new_q[i] /= norm

        self.q = new_q
        self.s = new_s
