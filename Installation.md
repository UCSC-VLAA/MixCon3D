## Installation

**The original instruction part is easy to follow but sometimes meets obstacles. Therefore, I list a more detailed one. Hope you find them well.**

If you would like to run the inference or (and) training locally, you may need to install the dependencies.

1. Create a conda environment 

```
conda create -n openshape python=3.9
conda activate openshape
```

2. Install [pytorch](https://pytorch.org/get-started/previous-versions/).

   I recommend using pip as the following line to install the pytorch (torch1.12.1, cuda11.3).

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

   The original command uses conda to install.
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

3. Install [MinkowskiEngine](https://nvidia.github.io/MinkowskiEngine/quick_start.html).
   The following command is easy to conduct but always meets errors.
```
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine
```
  Here are some more detailed and useful instructions:
  
  a) First, remember to check the version of libstdc++.so.6 using the following command.

```
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
```
  If you don't find "GLIBCXX_3.4.29" or more advanced required version, you need to upgrade the library ([Reference](https://stackoverflow.com/questions/65349875/where-can-i-find-glibcxx-3-4-29)).
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test # Ignore if not ubuntu
sudo apt-get update
# sudo apt-get install gcc-4.9 Please check the version of gcc !!!
sudo apt-get upgrade libstdc++6
```
  Then you can find the required "GLIBCXX_3.4.29".
  
  b) Second, I recommend installing the MinkowskiEngine locally.
```
git clone https://github.com/NVIDIA/MinkowskiEngine
cd MinkowskiEngine
```
  The following commands are recommened by this [issue](https://github.com/NVIDIA/MinkowskiEngine/issues/300).
  
```
conda install openblas-devel -c anaconda
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```
   Tip: Sometimes, an error about missing headers of pybind11 occurs. If so, using the following command to install pybind ([Reference](https://pybind11.readthedocs.io/en/stable/installing.html#)).

```
conda install -c conda-forge pybind11
```
   
4. Install the [DGL](https://www.dgl.ai/pages/start.html) by the following command.
```
conda install -c dglteam/label/cu113 dgl
```
5. Install the following packages:
```
pip install huggingface_hub wandb omegaconf torch_redstone einops tqdm open3d 
```
```
pip install open_clip_torch
```
**(opt.)** If you need downloading data (python download_data.py), you need to install chardet:
```
pip install chardet
```
**(opt.)** For CUDA version, if you find the current cuda version is not what you want, you can use the following commands to change the version by modifying the bashsrc file:
```
vim ~/.bashrc
```

```
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
```

```
source ~/.bashrc
```
**(opt.)** If the wandb meets problems like "AttributeError: partially initialized module 'charset_normalizer' has no attribute 'md__mypyc' (most likely due to a circular import)", the following command works:
```
pip install -U --force-reinstall charset-normalizer
```

**(opt.)** When installing the openblas (conda install openblas-devel -c anaconda), a problem sometimes occurs :
```
WARNING conda.gateways.disk.delete:unlink_or_rename_to_trash(143): Could not remove or rename /usr/local/anaconda3/pkgs/ca-certificates-2023.01.10-h06a4308_0/ssl/cacert.pem.  Please remove this file manually (you may need to reboot to free file handles)
```
This means the ca-certificates can not be updated. To delete these file, I chose to cd that folder and use "sudo rm -rf" to delete them. But, the NVML mismatch problem occurs after I delete them:

```
Failed to Initialize NVML: Driver / Library Version Mismatch problem solution
```
Here. I list how I solve this problem.

First, you can try to reboot this node and it sometimes works (from this [discussion](https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch/45319156#45319156)):
```
sudo reboot now
```
If rebooting not work, I turn to reinstall the GPU driver (following this [instruction](https://waydo.xyz/soft/linux/ubuntu-nvidia-apt/)). It works finally.

Before the above solution, I tried to reload the Nvidia driver (like [this](https://askubuntu.com/questions/1166317/module-nvidia-is-in-use-but-there-are-no-processes-running-on-the-gpu)), but it doesn't work. The reason, I guess, is the Nvidia loaded driver doesn't share between the below two commands:

```
cat /proc/driver/nvidia/version
```

```
dpkg -l | grep nvidia
```

