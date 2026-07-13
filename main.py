from huggingface_hub import hf_hub_download, upload_folder, upload_file
from tempfile import TemporaryDirectory
from flax import serialization

repo_id = "NiklasBli/hf-test"

def hf_upload_flax_parameter(parameter: dict, repo_id: str):
    with TemporaryDirectory() as tempdir:
        os_path = f"{str(tempdir)}/parameter.msgpack"
        with open(os_path, "wb") as file:
            packed = serialization.msgpack_serialize(parameter)
            file.write(packed)
        upload_file(path_or_fileobj=os_path, path_in_repo="parameter.msgpack", repo_id=repo_id, commit_message="uploade parameter")

def hf_download_flax_parameter(repo_id: str) -> dict:
    path = hf_hub_download(repo_id, "parameter.msgpack")
    with open(path, 'rb') as file:
        data = file.read()
        return serialization.msgpack_restore(data)

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

with open("config.json", "r") as f:
    config = json.load(f)

rbm_seed       = config["RBM config"]["seed"]
n_hidden_layer = config["RBM config"]["n_hidden_layer"]
bias           = config["RBM config"]["bias"]
param_init_std = config["RBM config"]["param_init_std"]

L = config["TFI Hamiltonian constants"]["L"]
J = config["TFI Hamiltonian constants"]["J"]
g = config["TFI Hamiltonian constants"]["g"]

mc_seed =  config["MC Sampler config"]["seed"]
n_therm =  config["MC Sampler config"]["n_therm"]
n_sample = config["MC Sampler config"]["n_sample"]

n_steps = config["Training config"]["n_steps"]

net = RBM(num_hidden=n_hidden_layer, bias=bias, param_init_std=param_init_std)
psi = jVMC.vqs.NQS(net, seed=rbm_seed)
psi.init_net(jnp.zeros((1,1,L)))
hamiltonian = make_TFI_hamiltonian(L, J, g)
sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(mc_seed), updateProposer=jVMC.sampler.propose_spin_flip_Z2, sweepSteps=L, numSamples=n_sample, thermalizationSweeps=n_therm)

train(hamiltonian, sampler, psi, L, g, n_steps)

hf_upload_flax_parameter(psi.parameters, repo_id)
upload_folder(folder_path="src", path_in_repo="src", repo_id=repo_id, allow_patterns="*.py")
upload_folder(folder_path="figures", path_in_repo="figures", repo_id=repo_id, allow_patterns="*.pdf")
upload_file(path_or_fileobj="config.json", path_in_repo="config.json", repo_id=repo_id)