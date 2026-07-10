import jax
import jax.random as random
import jax.numpy as jnp
import flax.linen as nn
import jVMC.sampler as sampler
from einops import einsum
import json

with open('src/config.json', 'r') as f:
    config = json.load(f)
float_dtype = jnp.dtype(config["float-dtype"])
int_dtype = jnp.dtype(config["int-dtype"])

def print_probability_distribution(psi: nn.Module, params: dict) -> jnp.ndarray:
    s = jnp.array(jnp.meshgrid(*[jnp.array([-1, 1])] * psi.L)).T.reshape(-1, psi.L)
    prob_dis = jax.vmap(lambda s: jnp.abs(psi.apply(params, s)) ** 2)(s)
    prob_dis /= jnp.sum(prob_dis)
    for i in range(prob_dis.shape[0]):
        print(f"State: {s[i]}, Probability: {prob_dis[i]}")

def print_histogram(samples: jnp.ndarray) -> dict:
    states = jnp.array(jnp.meshgrid(*[jnp.array([-1, 1])] * samples.shape[-1])).T.reshape(-1, samples.shape[-1])
    dict_dips = {}
    for s in states:
        dict_dips[str(s)] = 0
    for sample in samples:
        dict_dips[str(sample)] += 1 / len(samples)
    for s in states:
        print(f"State: {s}, Probability: {dict_dips[str(s)]}")

class TFI_Hamiltonian:

    def __init__(self, L: int, g: float, J: float=1.0):
        self.L = L
        self.g = g
        self.J = J
    
    def matrix_elements(self, s: jnp.ndarray) -> jnp.ndarray | jnp.ndarray:
        
        assert len(s.shape) in [1, 2], "Input array must be 1D or 2D."

        if len(s.shape) == 1:
            mat_els = jnp.zeros((self.L + 1,), dtype=float_dtype)
            s_primes = jnp.zeros((self.L + 1, self.L), dtype=int_dtype)
            s_primes[0] = s
            mat_els[0] = -self.J * jnp.sum(s * jnp.roll(s, shift=1, axis=-1), axis=-1)
    
            for i in range(1, self.L):
                s_primes[i] = s.at[i - 1].set(-s[i - 1])
                mat_els[i] = -self.g

        if len(s.shape) == 2:
            mat_els = jnp.zeros((s.shape[0], self.L + 1), dtype=float_dtype)
            s_primes = jnp.zeros((s.shape[0], self.L + 1, self.L), dtype=int_dtype)
            s_primes = s_primes.at[:, 0].set(s)
            mat_els = mat_els.at[:, 0].set(-self.J * jnp.sum(s * jnp.roll(s, shift=1, axis=-1), axis=-1))
    
            for i in range(1, self.L):
                s_primes = s_primes.at[:, i].set(s.at[:, i - 1].set(-s[:, i - 1]))
                mat_els = mat_els.at[:, i].set(-self.g)

        return mat_els, s_primes
    
    def energy(self, s: jnp.ndarray, psi: nn.Module, params: dict) -> jnp.ndarray:
        mat_els, s_primes = self.matrix_elements(s) # mat_els: b x L+1, s_primes: b x L+1 x L
        psi_s = psi.apply(params, s)  # b
        psi_s_primes = psi.apply(params, s_primes) # b x L+1
        if s.ndim == 1:
            return jnp.dot(mat_els, psi_s_primes) / psi_s
        else:
            return einsum(mat_els, psi_s_primes, "b n, b n -> b") / psi_s




class MCSampler:
    def __init__(
            self,
            L: int,
            rng: jax.Array
    ):
        self.L = L
        self.rng = rng
    
    def __call__(
            self,
            psi: nn.Module,
            params: dict,
            N: int,
            N_therm: int=20
    ) -> jnp.ndarray:
        
        samples = jnp.zeros((N, self.L), dtype=int_dtype)
        state = jnp.ones((self.L,), dtype=int_dtype)
        self.rng, sub_rng = random.split(self.rng)
        p_array = random.uniform(sub_rng, (self.L, N + N_therm))
        self.rng, sub_rng = random.split(self.rng)
        l_array = random.randint(sub_rng, (self.L, N + N_therm), 0, self.L-1)

        for i in range(N_therm):
            state = self.sweep(psi, params, p_array[i], l_array[i], state)

        for i in range(N):
            state = self.sweep(psi, params, p_array[i+N_therm], l_array[i+N_therm], state)
            samples = samples.at[i].set(state)
        return samples
    
    def sweep(self, psi, params, ps, ls, state):
        for l in range(self.L):
            state = self._update(psi, params, ps[l], ls[l], state)
        return state.at[:].set(state)

    def _update(self, psi, params, p, l, state):
        a = jnp.abs(psi.apply(params, state.at[l].set(-state[l])))**2 / jnp.abs(psi.apply(params, state)) ** 2
        if p < a:
            return state.at[l].set(-state[l])
        else:
            return state.at[:].set(state)
