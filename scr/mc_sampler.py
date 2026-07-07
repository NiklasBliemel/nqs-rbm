import jax
import jax.numpy as jnp
from flax import linen as nn


class MCSampler:
    def __init__(
            self,
            L: int,
            rng: jax.Array
    ):
        self.L = L
        self.rng = rng
        self.state: jnp.ndarray
    
    def __call__(
            self,
            psi: nn.Module,
            N: int, N_therm: int=20
    ) -> jnp.ndarray:
        samples = jnp.zeros((N, self.L), dtype=jnp.int8)
        self.state = jnp.ones((self.L,), dtype=jnp.int8)

        for _ in range(N_therm):
            self.sweep(psi)

        for i in range(N):
            self.sweep(psi)
            samples[i] = self.state
    
    def sweep(self, psi):
        for l in range(self.L):
            self._update(psi, l)

    def _update(self, psi, l):
        p = min(1, jnp.abs(psi(self.state.at[l].set(-self.state[l])) / psi(self.state)) ** 2)
        if jax.random.uniform(self.rng) < p:
            self.state.at[l].set(-self.state[l])
