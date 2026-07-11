import jax.numpy as jnp
from flax import linen as nn
import json
from huggingface_hub import ModelHubMixin

with open('src/config.json', 'r') as f:
    config = json.load(f)
float_dtype = jnp.dtype(config["float-dtype"])
param_init_std = config["param_init_std"]

class RBM(nn.Module, ModelHubMixin):

    num_hidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        layer = nn.Dense(features=self.num_hidden, use_bias=self.bias,
                            dtype=float_dtype,
                            kernel_init=nn.initializers.normal(param_init_std),
                            bias_init=nn.initializers.normal(param_init_std)
                        )
        h = jnp.sum(jnp.log(jnp.cosh(layer(2*s-1))), axis=-1)
        return h
    
    def _save_pretrained(self, save_directory):
        return super()._save_pretrained(save_directory)
    
    @classmethod
    def _from_pretrained(cls, *, model_id, revision, cache_dir, force_download, local_files_only, token, **model_kwargs):
        return super()._from_pretrained(model_id=model_id, revision=revision, cache_dir=cache_dir, force_download=force_download, local_files_only=local_files_only, token=token, **model_kwargs)