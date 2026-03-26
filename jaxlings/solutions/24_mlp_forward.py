"""Solution 24: Compose a 2-layer MLP forward pass."""
import jax
import jax.numpy as jnp


def run_exercise(x, params):
    w1, b1, w2, b2 = params
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2


if __name__ == "__main__":
    print(run_exercise)
