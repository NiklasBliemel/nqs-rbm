import jax
import jax.numpy as jnp
from flax import linen as nn
from .hamiltonian import TFI_Hamiltonian

def energy(H: TFI_Hamiltonian, psi: nn.Module, sample: jnp.ndarray) -> jnp.ndarray:
    E_loc = H.E_loc(sample, psi)
    return jnp.mean(E_loc)