from numbers import Real
import jax.numpy as jnp

def quad_loss(x: jnp.ndarray, u: jnp.ndarray) -> Real:
    """
    Quadratic loss.

    Args:
        x (jnp.ndarray):
        u (jnp.ndarray):

    Returns:
        Real
    """
    return jnp.sum(x.T @ x + u.T @ u)


def lifetime(x, lower_bound):
    l = 4
    while x % 2 == 0:
        l *= 2
        x /= 2

    return max(lower_bound, l + 1)
