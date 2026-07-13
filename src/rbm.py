import jax.numpy as jnp
from flax import linen as nn
import json

with open('config.json', 'r') as f:
    config = json.load(f)
float_dtype = jnp.dtype(config["Global defs"]["float-dtype"])

class RBM(nn.Module):

    num_hidden: int
    bias: bool
    param_init_std: float = 1e-2

    @nn.compact
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        layer = nn.Dense(features=self.num_hidden, use_bias=self.bias,
                            dtype=float_dtype,
                            kernel_init=nn.initializers.normal(self.param_init_std),
                            bias_init=nn.initializers.normal(self.param_init_std)
                        )
        h = jnp.sum(jnp.log(jnp.cosh(layer(2*s-1))), axis=-1)
        return h
