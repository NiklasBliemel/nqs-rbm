import jax
jax.config.update("jax_enable_x64", True)
import jax.random as random
import jVMC
import numpy as np
import matplotlib.pyplot as plt
import json

from nqs_network import RBM
from hamiltonian import make_TFI_hamiltonian

with open("src/config.json", "r") as f:
    config = json.load(f)

nn_seed = config["nn_seed"]
n_hidden_layer = config["n_hidden_layer"]

L = config["L"]
J = config["J"]
g = config["g"]

mc_seed = config["mc_seed"]
n_therm = config["n_therm"]
n_sample = config["n_sample"]
n_steps = config["n_steps"]

net = RBM(num_hidden=n_hidden_layer, bias=True)
psi = jVMC.vqs.NQS(net, seed=nn_seed)
hamiltonian = make_TFI_hamiltonian(L, J, g)

def energy_single_p_mode(h_t, P):
    return np.sqrt(1 + h_t**2 - 2 * h_t * np.cos(P))

def ground_state_energy_per_site(h_t, N):
    Ps = 0.5 * np.arange(- (N - 1), N - 1 + 2, 2)
    Ps = Ps * 2 * np.pi / N
    energies_p_modes = np.array([energy_single_p_mode(h_t, P) for P in Ps])
    return - 1 / N * np.sum(energies_p_modes)


exact_energy = ground_state_energy_per_site(g, L) # J=1.0
print(f"exact ground state energy: {exact_energy}")

sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(mc_seed), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=1, sweepSteps=L,
                                 numSamples=n_sample, thermalizationSweeps=n_therm)

# Set up TDVP
tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1., diagonalShift=10, makeReal='real')

stepper = jVMC.util.stepper.Euler(timeStep=1e-2)  # ODE integrator

res = []
for n in range(n_steps):

    dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=None)
    psi.set_parameters(dp)
    E_mean = jax.numpy.real(tdvpEquation.ElocMean0) / L
    E_var = tdvpEquation.ElocVar0 / L

    print(f"Step: {n:4d}\tE_mean: {E_mean:.4f}\t\tE_var: {E_var:.4f}")

    res.append([n, jax.numpy.real(tdvpEquation.ElocMean0) / L, tdvpEquation.ElocVar0 / L])

res = np.array(res)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=[4.8, 4.8])
ax[0].semilogy(res[:, 0], res[:, 1] - exact_energy, '-', label=r"$L=" + str(L) + "$")
ax[0].set_ylabel(r'$(E-E_0)/L$')

ax[1].semilogy(res[:, 0], res[:, 2], '-')
ax[1].set_ylabel(r'Var$(E)/L$')
ax[0].legend()
plt.xlabel('iteration')
plt.tight_layout()
plt.savefig('gs_search.pdf')