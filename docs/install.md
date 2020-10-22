# Installation steps
The procedure was tested on Ubuntu 18.04 with an installed CUDA driver of
version 10.1.
## Environmental variables:
* `$PD_MESH_NET_ROOT`: Root of this folder.
* `$PYG_ROOT`: Root of PyTorch Geometric.

## Steps
It will be assumed that [`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/)
is installed. Alternatively, the creation and activation of the virtualenv
(`mkvirtualenv` and `workon` in the instructions below) can be performed
according to the [official PyPA instructions](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
### Create a virtualenv:
```bash
mkvirtualenv pd_mesh_net --python=$(which python3)
```
### Install PyTorch Geometric:
```bash
export THIRD_PARTY_ROOT=$PD_MESH_NET_ROOT/third_party
export PYG_ROOT=$THIRD_PARTY_ROOT/pytorch_geometric
```
- Clone PyTorch Geometric:
    ```bash
    cd $THIRD_PARTY_ROOT
    git clone -n https://github.com/rusty1s/pytorch_geometric.git
    cd $PYG_ROOT
    git checkout 082fb83
    ```
- If not already in `~/.bashrc`, add CUDA to `$PATH` and `$CPATH`:
    ```bash
    export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
    export CPATH=/usr/local/cuda-10.1/include:$CPATH
    ```
- Add CUDA to `$LD_LIBRARY_PATH`:
    ```bash
    export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
    ```
- Install dependencies (in virtualenv, i.e., run `workon pd_mesh_net` first):
    ```bash
    pip install torch==1.4.0
    ```
    - Installed `torch-1.4.0` (https://files.pythonhosted.org/packages/24/19/4804aea17cd136f1705a5e98a00618cb8f6ccc375ad8bfa437408e09d058/torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl)
    ```bash
    pip install numpy
    ```
    - Installed `numpy-1.18.1` (https://files.pythonhosted.org/packages/62/20/4d43e141b5bc426ba38274933ef8e76e85c7adea2c321ecf9ebf7421cedf/numpy-1.18.1-cp36-cp36m-manylinux1_x86_64.whl)
    ```bash
    pip install --verbose --no-cache-dir torch-scatter==1.4.0
    ```
    - Installed `torch-scatter-1.4.0` (https://files.pythonhosted.org/packages/b8/c3/8bad887ffa55c86f120ef5ae252dc0e357b3bd956d9fbf45242bacc46290/torch_scatter-1.4.0.tar.gz)
    ```bash
    pip install --verbose --no-cache-dir torch-sparse==0.4.3
    ```
    - Installed `scipy-1.4.1` (https://files.pythonhosted.org/packages/dc/29/162476fd44203116e7980cfbd9352eef9db37c49445d1fec35509022f6aa/scipy-1.4.1-cp36-cp36m-manylinux1_x86_64.whl)
    - Installed `torch-sparse-0.4.3` (https://files.pythonhosted.org/packages/08/4e/a268613fa6a92ffbc65b89e66fc8be5590801937185007f0f7bcb75ea21f/torch_sparse-0.4.3.tar.gz)
    ```bash
    pip install --verbose --no-cache-dir torch-cluster==1.4.5
    ```
    - Installed `torch_cluster-1.4.5` (https://files.pythonhosted.org/packages/c3/70/1d827d6fd1e03bb5ae84852dd0070c6574105c37e7b935284f6e990932db/torch_cluster-1.4.5.tar.gz)
    ```bash
    pip install --verbose --no-cache-dir torch-spline-conv==1.1.1
    ```
    - Installed `torch_spline_conv-1.1.1` (https://files.pythonhosted.org/packages/5e/77/5420584cdb1514c580722ca4bc482a509105d64b7c70246e9dc4a3e6d3c5/torch_spline_conv-1.1.1.tar.gz)

- Install PyTorch Geometric:
    ```bash
    cd $PYG_ROOT
    ```
    - Comment lines 30 and 31 in `$PYG_ROOT/setup.py` (i.e., lines starting for `url` and `download_url`).
    ```bash
    workon pd_mesh_net
    python setup.py install
    ```
- Install optional PyTorch-Geometric dependencies:
    ```bash
    pip install tensorboard
    ```
    As a reference, the above command installed the following package versions:
    - Installed `absl-py-0.9.0` (https://files.pythonhosted.org/packages/1a/53/9243c600e047bd4c3df9e69cfabc1e8004a82cac2e0c484580a78a94ba2a/absl-py-0.9.0.tar.gz)
    - Installed `cachetools-4.0.0` (https://files.pythonhosted.org/packages/08/6a/abf83cb951617793fd49c98cb9456860f5df66ff89883c8660aa0672d425/cachetools-4.0.0-py3-none-any.whl)
    - Installed `google-auth-1.11.0` (https://files.pythonhosted.org/packages/1c/6d/7aae38a9022f982cf8167775c7fc299f203417b698c27080ce09060bba07/google_auth-1.11.0-py2.py3-none-any.whl)
    - Installed `google-auth-oauthlib-0.4.1` (https://files.pythonhosted.org/packages/7b/b8/88def36e74bee9fce511c9519571f4e485e890093ab7442284f4ffaef60b/google_auth_oauthlib-0.4.1-py2.py3-none-any.whl)
    - Installed `grpcio-1.26.0` (https://files.pythonhosted.org/packages/17/8f/f79c5c174bebece41f824dd7b1ba98da45dc2d4c373b38ac6a7f6a5acb5e/grpcio-1.26.0-cp36-cp36m-manylinux2010_x86_64.whl)
    - Installed `markdown-3.1.1` (https://files.pythonhosted.org/packages/c0/4e/fd492e91abdc2d2fcb70ef453064d980688762079397f779758e055f6575/Markdown-3.1.1-py2.py3-none-any.whl)
    - Installed `oauthlib-3.1.0` (https://files.pythonhosted.org/packages/05/57/ce2e7a8fa7c0afb54a0581b14a65b56e62b5759dbc98e80627142b8a3704/oauthlib-3.1.0-py2.py3-none-any.whl)
    - Installed `protobuf-3.11.2` (https://files.pythonhosted.org/packages/ca/ac/838c8c8a5f33a58132dd2ad2a30329f6ae1614a9f56ffb79eaaf71a9d156/protobuf-3.11.2-cp36-cp36m-manylinux1_x86_64.whl)
    - Installed `pyasn1-0.4.8` (https://files.pythonhosted.org/packages/62/1e/a94a8d635fa3ce4cfc7f506003548d0a2447ae76fd5ca53932970fe3053f/pyasn1-0.4.8-py2.py3-none-any.whl)
    - Installed `pyasn1-modules-0.2.8` (https://files.pythonhosted.org/packages/95/de/214830a981892a3e286c3794f41ae67a4495df1108c3da8a9f62159b9a9d/pyasn1_modules-0.2.8-py2.py3-none-any.whl)
    - Installed `requests-oauthlib-1.3.0` (https://files.pythonhosted.org/packages/a3/12/b92740d845ab62ea4edf04d2f4164d82532b5a0b03836d4d4e71c6f3d379/requests_oauthlib-1.3.0-py2.py3-none-any.whl)
    - Installed `rsa-4.0` (https://files.pythonhosted.org/packages/02/e5/38518af393f7c214357079ce67a317307936896e961e35450b70fad2a9cf/rsa-4.0-py2.py3-none-any.whl)
    - Installed `tensorboard-2.1.0` (https://files.pythonhosted.org/packages/40/23/53ffe290341cd0855d595b0a2e7485932f473798af173bbe3a584b99bb06/tensorboard-2.1.0-py3-none-any.whl)
    - Installed `werkzeug-0.16.0` (https://files.pythonhosted.org/packages/ce/42/3aeda98f96e85fd26180534d36570e4d18108d62ae36f87694b476b83d6f/Werkzeug-0.16.0-py2.py3-none-any.whl)
### Install PyMesh
- Make sure your version of CMake is >= 3.11 (follow, e.g., https://answers.ros.org/question/293119/how-can-i-updateremove-cmake-without-partially-deleting-my-ros-distribution/?answer=297523#post-id-297523 if this is not the case).
- Install PyMesh from the Github repo:
    ```bash
    workon pd_mesh_net
    pip install git+git://github.com/PyMesh/PyMesh.git@93d182c0a7cee446e89fac74033347d900054af4
    ```
### Install other dependencies
```bash
pip install pyyaml
```
As a reference, the above command installed the following package versions:
- Installed `pyyaml-5.3`
### Install package
```bash
cd $PD_MESH_NET_ROOT
workon pd_mesh_net
pip install -e .
```
