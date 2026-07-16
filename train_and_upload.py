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
from hg_flax_helperfunctions import hf_upload_flax_parameter
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
private = config["Global defs"]["hf_private_repo"]
revision = f"L{L}_g{g}"
parameter_save_dtype = config["Global defs"]["parameter_save_dtype"]

# --- setup HF repository ---
login() # returns None if already logged in
try:
    create_repo(repo_id, repo_type="model", private=private)
    print(f"Repository {repo_id} created.")
except Exception:
    print(f"Repository {repo_id} already exists.")

# --- upload source files and requirements to main branch ---
upload_folder(folder_path="src", path_in_repo="src", repo_id=repo_id, allow_patterns="*.py")
upload_file(path_or_fileobj="pyproject.toml", path_in_repo="pyproject.toml", repo_id=repo_id)

# --- create branch for model configuration ---
try:
    create_branch(repo_id=repo_id, repo_type="model", branch=revision)
    print(f"Branch '{revision}' created.")
except Exception:
    print(f"Branch '{revision}' already exists. Proceeding to upload.")

# --- model setup & training ---
net = RBM(num_hidden=n_hidden_layer, bias=bias, param_init_std=param_init_std)
psi = jVMC.vqs.NQS(net, seed=rbm_seed)
psi.init_net(jnp.zeros((1,1,L)))
hamiltonian = make_TFI_hamiltonian(L, g)
sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(mc_seed), updateProposer=jVMC.sampler.propose_spin_flip_Z2, sweepSteps=L, numSamples=n_sample, thermalizationSweeps=n_therm)
train(hamiltonian, sampler, psi, L, g, n_steps)

# --- upload config, parameters and visualizations to corresponding branch ---
hf_upload_flax_parameter(psi.parameters, repo_id=repo_id, revision=revision, dtype=parameter_save_dtype)
upload_file(path_or_fileobj="config.json", path_in_repo="config.json", repo_id=repo_id, revision=revision)
upload_folder(folder_path="figures", path_in_repo="figures", repo_id=repo_id, revision=revision, allow_patterns="*.pdf")