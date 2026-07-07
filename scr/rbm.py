import jax.numpy as jnp
from flax import linen as nn


class RBM(nn.Module):
    L: int
    M: int

    def setup(self):
        self.linear = nn.Dense(
            features=(self.M, self.L),
            use_bias=True,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64
            )
    
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        assert s.shape[-1] == self.L, f"Size of spin-array(s) {s.shape[-1]} does not match expected Size {self.L}"
        h = jnp.prod(jnp.cosh(self.linear(s)), axis=-1)
        return h