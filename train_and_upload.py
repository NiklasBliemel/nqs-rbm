from huggingface_hub import upload_folder, upload_file, login, create_branch, create_repo
from tempfile import TemporaryDirectory
from flax import serialization
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
import jVMC
import json

from src.rbm import RBM
from src.hamiltonian import make_TFI_hamiltonian
from src.train import train
import json


# --- loading experiment config ---
with open("config.json", "r") as f:
    config = json.load(f)

rbm_seed       = config["RBM config"]["seed"]
n_hidden_layer = config["RBM config"]["n_hidden_layer"]
bias           = config["RBM config"]["bias"]
param_init_std = config["RBM config"]["param_init_std"]

L = config["TFI Hamiltonian constants"]["L"]
g = config["TFI Hamiltonian constants"]["g"]

mc_seed =  config["MC Sampler config"]["seed"]
n_therm =  config["MC Sampler config"]["n_therm"]
n_sample = config["MC Sampler config"]["n_sample"]

n_steps = config["Training config"]["n_steps"]

# --- HF configuration ---
repo_id = "NiklasBli/nqs-rbm"
private = False
config_branch = f"rbm_{L}_{n_hidden_layer}_{g}"

# --- setup HF repository & branch if not already existing---
login() # returns None if already logged in
try:
    create_repo(repo_id, repo_type="model", private=private)
    print(f"Repository {repo_id} created.")
except Exception:
    print(f"Repository {repo_id} already exists.")
try:
    create_branch(repo_id=repo_id, repo_type="model", branch=config_branch)
    print(f"Branch '{config_branch}' created.")
except Exception:
    print(f"Branch '{config_branch}' already exists. Proceeding to upload.")

# --- model setup & training ---
net = RBM(num_hidden=n_hidden_layer, bias=bias, param_init_std=param_init_std)
psi = jVMC.vqs.NQS(net, seed=rbm_seed)
psi.init_net(jnp.zeros((1,1,L)))
hamiltonian = make_TFI_hamiltonian(L, g)
sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(mc_seed), updateProposer=jVMC.sampler.propose_spin_flip_Z2, sweepSteps=L, numSamples=n_sample, thermalizationSweeps=n_therm)
train(hamiltonian, sampler, psi, L, g, n_steps)

# --- upload to HF ---
# config branch
with TemporaryDirectory() as tempdir:
    os_path = f"{str(tempdir)}/parameter.msgpack"
    with open(os_path, "wb") as file:
        packed = serialization.msgpack_serialize(psi.parameters)
        file.write(packed)
    upload_file(path_or_fileobj=os_path, path_in_repo="parameter.msgpack", repo_id=repo_id, revision=config_branch)
upload_file(path_or_fileobj="config.json", path_in_repo="config.json", repo_id=repo_id, revision=config_branch)
upload_folder(folder_path="figures", path_in_repo="figures", repo_id=repo_id, revision=config_branch, allow_patterns="*.pdf")
# main branch
upload_folder(folder_path="src", path_in_repo="src", repo_id=repo_id, allow_patterns="*.py")
upload_file(path_or_fileobj="pyproject.toml", path_in_repo="pyproject.toml", repo_id=repo_id)