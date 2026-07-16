# NQS - Restricted Boltzmann Machine (RBM)

This is a example project for uploading a jVMC flax model to huggingface.co <br>

- **conected huggingface repo**: (https://huggingface.co/NiklasBli/hf-test)
- **hf_flax_helperfunctions.py**: simple functions to download and upload flax parameter pyTrees
- **train_and_upload.py:** production code training a model defined by src/ and config.jason and uploading it to huggingface.co using following convention:
  - **main Branch:** requirements and source code (model architecture, hamiltonian and training scheme)
  - **other Branches:** parameters of trained model, used configuration and visualization (training graphs etc.)
    - _naming-convention:_ rbm\_{number_of_sites_L}\_{number_of_hidden_layers}\_{transverse_field_strength_g}

## Transverse Field Ising (TFI) Hamiltonian:

For this hamiltonian the RBM is searching the groundstate.

$$
H = -\sum_{i=1}^L \hat S_{z,i} \hat S_{z,i+1} - g \sum_{i=1}^L \hat S_{x,i}
$$
