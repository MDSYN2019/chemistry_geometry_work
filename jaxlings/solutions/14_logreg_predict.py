"""Solution 14: Build logistic predictions from logits."""
import jax
import jax.numpy as jnp


def run_exercise(x, w, b):
    logits = x @ w + b
    return jax.nn.sigmoid(logits)


if __name__ == "__main__":
    print(run_exercise)
