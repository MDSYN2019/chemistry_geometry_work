"""Solution 12: Implement a dense layer forward pass."""
import jax
import jax.numpy as jnp


def run_exercise(x, w, b):
    return x @ w + b


if __name__ == "__main__":
    print(run_exercise)
