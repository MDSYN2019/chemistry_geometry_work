"""Solution 03: Compute sums, means, and argmax reductions."""
import jax
import jax.numpy as jnp


def run_exercise():
    x = jnp.array([[1., 4., 2.], [7., 0., 5.]])
    return {
        "sum": jnp.sum(x),
        "row_mean": jnp.mean(x, axis=1),
        "argmax": jnp.argmax(x),
    }


if __name__ == "__main__":
    print(run_exercise)
