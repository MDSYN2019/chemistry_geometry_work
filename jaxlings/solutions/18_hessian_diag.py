"""Solution 18: Compute Hessian diagonal via nested grads."""
import jax
import jax.numpy as jnp


def run_exercise(x):
    f = lambda t: jnp.sum(t**3)
    hess = jax.jacfwd(jax.jacrev(f))(x)
    return jnp.diag(hess)


if __name__ == "__main__":
    print(run_exercise)
