from huggingface_hub import hf_hub_download, upload_folder, upload_file, login, create_branch, create_repo
from tempfile import TemporaryDirectory
from flax import serialization
import h5py


def hf_upload_flax_parameter(parameter: dict, repo_id: str, branch_name: str = "main", dtype: str = "msgpack"):
    '''''
    dtype options: "msgpack", "hdf5" / "h5"
    '''''
    with TemporaryDirectory() as tempdir:
        os_path = f"{str(tempdir)}/parameter.{dtype}"
        if dtype == "msgpack":
            with open(os_path, "wb") as file:
                packed = serialization.msgpack_serialize(parameter)
                file.write(packed)
        elif dtype in ["hdf5", "h5"]:
            save_dict_to_hdf5(os_path, parameter)
        else:
            print("Error: Data type not available! (Choose msgpack or hdf5)")
            return 0
        upload_file(path_or_fileobj=os_path, path_in_repo=f"parameter.{dtype}", repo_id=repo_id, revision=branch_name)


def hf_download_flax_parameter(repo_id: str, model_branch: str = "main", dtype: str = "msgpack") -> dict:
    '''''
    Example:
    parameter = hf_download_flax_parameter("NiklasBli/nqs-rbm", "rbm_10_6_0.7")
    '''''
    path = hf_hub_download(repo_id=repo_id, revision=model_branch, filename=f"parameter.{dtype}")
    if dtype == "msgpack":
        with open(path, 'rb') as file:
            data = file.read()
            return serialization.msgpack_restore(data)
    elif dtype in ["hdf5", "h5"]:
        return load_dict_from_hdf5(path)
    else:
        print("Error: Data type not available! (Choose msgpack or hdf5)")
        return 0

def save_dict_to_hdf5(path: str, data: dict):
    with h5py.File(path, "w") as file:
        _save_dict_to_hdf5(file, "", data)

def load_dict_from_hdf5(path: str) -> dict:
    with h5py.File(path, "r") as file:
        out = _save_dict_to_hdf5(file, "")
    return out

def _save_dict_to_hdf5(h5_file, path, dic):
    for key, item in dic.items():
        subpath = f"{path}/{key}"
        if isinstance(item, dict):
            _save_dict_to_hdf5(h5_file, subpath, item)
        else:
            h5_file.create_dataset(subpath, data=item)

def _load_dict_from_hdf5(h5_file, path):
    dic = {}
    for key, item in h5_file[path].items():
        subpath = f"{path}/{key}"
        if isinstance(item, h5py.Group):
            dic[key] = _load_dict_from_hdf5(h5_file, subpath)
        else:
            dic[key] = item[()]
    return dic