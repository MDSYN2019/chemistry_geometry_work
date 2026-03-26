"""Solution 15: Implement stable softmax cross-entropy."""
import jax
import jax.numpy as jnp


def run_exercise(logits, labels):
    log_probs = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.mean(jnp.sum(labels * log_probs, axis=-1))


if __name__ == "__main__":
    print(run_exercise)
