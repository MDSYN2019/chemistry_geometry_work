"""Solution 02: Use broadcasting for row/column transforms."""
import jax
import jax.numpy as jnp


def run_exercise():
    x = jnp.arange(6.).reshape(2, 3)
    row_bias = jnp.array([1., 2., 3.])
    col_scale = jnp.array([[2.], [3.]])
    return (x + row_bias) * col_scale


if __name__ == "__main__":
    print(run_exercise)
