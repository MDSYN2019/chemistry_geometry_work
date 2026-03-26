"""Solution 13: Write one SGD update step."""
import jax
import jax.numpy as jnp


def run_exercise(w, x, y, lr=0.1):
    def loss_fn(param):
        pred = x * param
        return jnp.mean((pred - y) ** 2)
    loss, grad = jax.value_and_grad(loss_fn)(w)
    new_w = w - lr * grad
    return loss, new_w


if __name__ == "__main__":
    print(run_exercise)
