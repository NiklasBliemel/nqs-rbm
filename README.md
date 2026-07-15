# NQS - Restricted Boltzmann Machine (RBM)

This is a example project for uploading a jVMC flax model to huggingface.co <br>

- **conected huggingface repo**: (https://huggingface.co/NiklasBli/hf-test)
- **hf_flax_helperfunctions.py**: simple functions to download and upload flax parameter pyTrees
- **train_and_upload.py:** production code training a model defined by src/ and config.jason and uploading it to huggingface.co using following convention:
  - **main Branch:** requirements and source code (model architecture, hamiltonian and training scheme)
  - **other Branches:** parameters of trained model, used configuration and visualization (training graphs etc.)
    - _naming-convention:_ rbm\_{number*of_sites*(L)}\_{number*of_hidden_layers}\_{transvers_field_strength*(g)}
