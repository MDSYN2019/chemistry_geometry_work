"""Solution 23: Save/load parameter pytree leaves via numpy."""
import jax
import jax.numpy as jnp

from pathlib import Path
import numpy as np

def run_exercise(params, out_file='jax_params.npz'):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    arrays = {f'leaf_{i}': np.asarray(v) for i, v in enumerate(leaves)}
    np.savez(out_file, **arrays)
    loaded = np.load(out_file)
    restored_leaves = [jnp.asarray(loaded[f'leaf_{i}']) for i in range(len(leaves))]
    return jax.tree_util.tree_unflatten(treedef, restored_leaves)


if __name__ == "__main__":
    print(run_exercise)
