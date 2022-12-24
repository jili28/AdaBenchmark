from numbers import Real

import gymnasium as gym
import jax.numpy as jnp
import numpy as np


class Pendulum:

    def __init__(self, g=2, render=False, states='angular'):
        # render_mode = 'human'
        if render:
            self.env = gym.make('Pendulum-v1', render_mode='human', g=g, max_episode_steps=2000)
        else:
            self.env = gym.make('Pendulum-v1', g=g, max_episode_steps=2000)

        # self.Q = jnp.array([[1, 0], [0, 0.1]])
        # self.R = jnp.array([0.001])

        # self.Q = jnp.array([[-1, 0], [0, -0.1]])
        self.R = jnp.array([-0.001])

        self.d_a = 1
        if states == 'angular':
            self.d_s = 2
            self.Q = jnp.array([[-1, 0], [0, -0.1]])
        elif states == 'full':
            self.Q = jnp.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, -0.1]])
            self.d_s = 4
        else:
            raise NotImplemented(f"{states} not an environment mode")
        self.reset()

    def reset(self):
        self.env.reset()

    def bounds(self):
        return [jnp.pi, 8]

    def get_cost_function(self):
        def quad_loss(x: jnp.ndarray, u: jnp.ndarray) -> Real:
            """
            Quadratic loss.

            Args:
                x (jnp.ndarray):
                u (jnp.ndarray):

            Returns:
                Real
            """
            return jnp.sum(x.T @ self.Q @ x + u.T @ self.R @ u)

        return quad_loss

    def render(self):
        self.env.render()

    def __call__(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # angle = (-1 if obs[0] <= 0 else 1) * jnp.arccos(obs[0])
        angle, _ = self.env.state
        angle = ((angle + np.pi) % (2 * np.pi)) - np.pi
        if self.d_s == 2:
            return jnp.array([angle, obs[2]]), reward, done, truncated, info
        elif self.d_s == 4:
            return jnp.array([obs[0], obs[1], angle, obs[2]]), reward, done, truncated, info
