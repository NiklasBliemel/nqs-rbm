from huggingface_hub import hf_hub_download, upload_folder, upload_file, login, create_branch, create_repo
from tempfile import TemporaryDirectory
from flax import serialization


def hf_upload_flax_parameter(parameter: dict, repo_id: str, branch_name: str = "main"):
    with TemporaryDirectory() as tempdir:
        os_path = f"{str(tempdir)}/parameter.msgpack"
        with open(os_path, "wb") as file:
            packed = serialization.msgpack_serialize(parameter)
            file.write(packed)
        upload_file(path_or_fileobj=os_path, path_in_repo="parameter.msgpack", repo_id=repo_id, revision=branch_name, commit_message="uploade parameter")

def hf_download_flax_parameter(repo_id: str, model_branch: str = "main") -> dict:
    '''''
    Example:
    parameter = hf_download_flax_parameter("NiklasBli/nqs-rbm", "rbm_10_6_0.7")
    '''''
    path = hf_hub_download(repo_id=repo_id, revision=model_branch, filename="parameter.msgpack")
    with open(path, 'rb') as file:
        data = file.read()
        return serialization.msgpack_restore(data)