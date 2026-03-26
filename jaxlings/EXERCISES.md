# Exercise List (with solution mapping)

| # | Track | Topic | Exercise File | Solution File |
|---|---|---|---|---|
| 00 | JAX | Create arrays, reshape, and stack with jax.numpy. | `exercises/00_jnp_arrays/array_basics.py` | `solutions/00_array_basics.py` |
| 01 | JAX | Practice immutable updates with `.at[...]`. | `exercises/01_jnp_indexing/slicing_and_updates.py` | `solutions/01_slicing_and_updates.py` |
| 02 | JAX | Use broadcasting for row/column transforms. | `exercises/02_broadcasting/broadcasting_ops.py` | `solutions/02_broadcasting_ops.py` |
| 03 | JAX | Compute sums, means, and argmax reductions. | `exercises/03_reductions/reductions.py` | `solutions/03_reductions.py` |
| 04 | JAX | Split PRNG keys and sample reproducibly. | `exercises/04_random/prng_keys.py` | `solutions/04_prng_keys.py` |
| 05 | JAX | Vectorize a scalar function with `jax.vmap`. | `exercises/05_vmap/vmap_basics.py` | `solutions/05_vmap_basics.py` |
| 06 | JAX | Differentiate a scalar-valued function with `jax.grad`. | `exercises/06_grad/grad_scalar.py` | `solutions/06_grad_scalar.py` |
| 07 | JAX | Return loss and gradient together. | `exercises/07_value_and_grad/value_and_grad_loss.py` | `solutions/07_value_and_grad_loss.py` |
| 08 | JAX | Wrap pure math with `jax.jit`. | `exercises/08_jit/jit_speed_stub.py` | `solutions/08_jit_speed_stub.py` |
| 09 | JAX | Use `lax.cond` for branchy tensor logic. | `exercises/09_lax_control_flow/lax_cond_where.py` | `solutions/09_lax_cond_where.py` |
| 10 | JAX | Build cumulative sums with `lax.scan`. | `exercises/10_lax_scan/scan_cumsum.py` | `solutions/10_scan_cumsum.py` |
| 11 | JAX | Map over nested parameter pytrees. | `exercises/11_pytree/pytree_params.py` | `solutions/11_pytree_params.py` |
| 12 | JAX | Implement a dense layer forward pass. | `exercises/12_nn_dense/dense_layer.py` | `solutions/12_dense_layer.py` |
| 13 | JAX | Write one SGD update step. | `exercises/13_training_step/sgd_step.py` | `solutions/13_sgd_step.py` |
| 14 | JAX | Build logistic predictions from logits. | `exercises/14_logistic_regression/logreg_predict.py` | `solutions/14_logreg_predict.py` |
| 15 | JAX | Implement stable softmax cross-entropy. | `exercises/15_softmax_cross_entropy/cross_entropy.py` | `solutions/15_cross_entropy.py` |
| 16 | JAX | Normalize a batch with epsilon stabilization. | `exercises/16_batchnorm_like/normalize_batch.py` | `solutions/16_normalize_batch.py` |
| 17 | JAX | Apply 1D valid convolution with `lax.conv_general_dilated`. | `exercises/17_convolution/conv1d_valid.py` | `solutions/17_conv1d_valid.py` |
| 18 | JAX | Compute Hessian diagonal via nested grads. | `exercises/18_autodiff_higher_order/hessian_diag.py` | `solutions/18_hessian_diag.py` |
| 19 | JAX | Combine MSE and L2 regularization. | `exercises/19_custom_loss/regularized_mse.py` | `solutions/19_regularized_mse.py` |
| 20 | JAX | Clip gradients to a max global norm. | `exercises/20_grad_clipping/clip_grad_norm.py` | `solutions/20_clip_grad_norm.py` |
| 21 | JAX | Compute Jacobian of a vector function. | `exercises/21_jacobian/jacobian_linear.py` | `solutions/21_jacobian_linear.py` |
| 22 | JAX | Read and complete a minimal `pmap` skeleton. | `exercises/22_pmap_reading/pmap_stub.py` | `solutions/22_pmap_stub.py` |
| 23 | JAX | Save/load parameter pytree leaves via numpy. | `exercises/23_checkpointing/serialize_params.py` | `solutions/23_serialize_params.py` |
| 24 | JAX | Compose a 2-layer MLP forward pass. | `exercises/24_capstone_mlp/mlp_forward.py` | `solutions/24_mlp_forward.py` |
