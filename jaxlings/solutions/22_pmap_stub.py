"""Solution 22: Read and complete a minimal `pmap` skeleton."""
import jax
import jax.numpy as jnp


def run_exercise(x):
    # This runs on one or many local devices.
    mapped = jax.pmap(lambda t: t * 2)
    return mapped(x)


if __name__ == "__main__":
    print(run_exercise)
