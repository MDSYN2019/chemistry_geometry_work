"""Solution 17: Apply 1D valid convolution with `lax.conv_general_dilated`."""
import jax
import jax.numpy as jnp

from jax import lax

def run_exercise(signal, kernel):
    signal4 = signal[None, :, None]
    kernel4 = kernel[:, None, None]
    out = lax.conv_general_dilated(
        signal4,
        kernel4,
        window_strides=(1,),
        padding='VALID',
        dimension_numbers=('NWC', 'WIO', 'NWC'),
    )
    return out[0, :, 0]


if __name__ == "__main__":
    print(run_exercise)
