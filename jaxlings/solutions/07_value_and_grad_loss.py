"""Solution 07: Return loss and gradient together."""
import jax
import jax.numpy as jnp


def run_exercise(w=1.5):
    loss = lambda t: (t - 3.0) ** 2
    value, grad = jax.value_and_grad(loss)(w)
    return value, grad


if __name__ == "__main__":
    print(run_exercise)
