import jax.numpy as jnp
from flax import linen as nn
import json

with open('src/config.json', 'r') as f:
    config = json.load(f)
float_dtype = jnp.dtype(config["float-dtype"])
int_dtype = jnp.dtype(config["int-dtype"])
std = config["std"]

class RBM(nn.Module):
    L: int
    M: int

    def setup(self):
        self.dense = nn.Dense(
            features=self.M,
            use_bias=True,
            dtype=float_dtype,
            kernel_init=nn.initializers.normal(std),
            bias_init=nn.initializers.normal(std),
        )
    
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        h = jnp.prod(jnp.cosh(self.dense(s)), axis=-1)
        return h