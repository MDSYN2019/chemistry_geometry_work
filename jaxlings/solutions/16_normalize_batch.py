"""Solution 16: Normalize a batch with epsilon stabilization."""
import jax
import jax.numpy as jnp


def run_exercise(x, eps=1e-5):
    mean = jnp.mean(x, axis=0, keepdims=True)
    var = jnp.var(x, axis=0, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


if __name__ == "__main__":
    print(run_exercise)
