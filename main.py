import json

import json
import jax
import jax.random as random
import jax.numpy as jnp
import flax.linen as nn

with open("src/config.json", "r") as f:
    config = json.load(f)

seed = config["seed"]
lr = config["learning_rate"]
L = config["L"]
M = config["M"]
J = config["J"]
g = config["g"]

N_therm = config["N_therm"]
N_mc = config["N_mc"]
n_epochs = config["n_epochs"]


def main():
    print("hello")
    return 0
    rng = jax.random.PRNGKey(42)
    rng, rng_init = jax.random.split(rng)
    mc_sampler = MCSampler(L=L, rng=rng)
    H = TFI_Hamiltonian(L=L, J=J, g=g)
    rbm = RBM(L=L, M=M)
    s = jnp.ones(L, dtype=jnp.int8)
    params = rbm.init(rng_init, s)
    optimizer = adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    samples = mc_sampler(psi=rbm, params=params, N=10, N_therm=0)
    print(samples)

    E = H.E_loc(samples, rbm, params)
    print(E)


if __name__ == "__main__":
    main()
