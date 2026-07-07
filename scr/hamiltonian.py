import jax.numpy as jnp
from flax import linen as nn


class TFI_Hamiltonian:

    def __init__(self, L: int, g: float, J: float=1.0):
        self.L = L
        self.g = g
        self.J = J
    
    def matrix_elements(self, s: jnp.ndarray) -> jnp.ndarray | jnp.ndarray:
        mat_els = jnp.zeros((self.L + 1,), dtype=jnp.float64)
        s_primes = jnp.zeros((self.L + 1, self.L), dtype=jnp.int8)
        s_primes[0] = s
        mat_els[0] = -self.J * jnp.sum(s * jnp.roll(s, shift=1, axis=-1), axis=-1)

        for i in range(1, self.L):
            s_primes[i] = s.at[i - 1].set(-s[i - 1])
            mat_els[i] = -self.g

        return mat_els, s_primes
    
    def E_loc(self, s: jnp.ndarray, psi: nn.Module) -> jnp.ndarray:
        mat_els, s_primes = self.matrix_elements(s)
        psi_s = psi(s)
        psi_s_primes = psi(s_primes)
        E_loc = jnp.sum(mat_els * (psi_s_primes), axis=-1) / psi_s
        return E_loc
        