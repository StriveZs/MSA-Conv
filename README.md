# MSA-Conv CUDA Version
pytorch implementation "TiC: Exploring Vision Transformer in Convolution", Self-Attention Meets Conv!

[Paper Link](https://arxiv.org/pdf/2310.04134.pdf)

Installation Guide for MSA-Conv

## Environment Setup
1. Docker container
We are utilizing CUDA Version 10.1 consequently, we have installed the cuda-pytorch:10.1-1.5 image to fulfill our machine's environmental requisites. It is worth noting that the installation of other CUDA versions can also be accomplished seamlessly and accurately.
```
docker run -it  --user root --gpus all --ipc=host --shm-size 8G  -e NVIDIA_VISIBLE_DEVICES=xxx --name="xxx"  -p xxx:22 -v xxx:/file nablascom/cuda-pytorch:10.1-1.5 /bin/bash
```

2. Minconda 
We utilize Anaconda for Python environment management. Our environment is based on Python 3.7, but higher or lower versions are also compatible.

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

conda create -y --name pytorch_use python=3.7
```

3. Pytorch
Torch==1.5.0 is not a stringent requirement; users can install the version that aligns with their needs.
```
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
```
4. pybind11
Pybind11 is a mandatory prerequisite for installing the MSA-Conv into the Python package
```
pip install "pybind11[global]" -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

## MSA-Conv Setup
Users may need to modify certain file paths in setup.py to align with their specific environment.
1. cmd installation
```
cd /../msa_conv_cuda/

python setup.py develop
```

## Usage
The validation file employs our MSA-Conv to assess and validate the accuracy of gradients obtained through backward propagation
