
1. Uninstall existing torch installations
pip3 uninstall torch torchvision torchaudio -y

1. Add NVIDIA Jetson's apt repository
sudo apt update
sudo apt install python3-pip libopenblas-base libopenmpi-dev

1. check jetson version 
sudo apt show nvidia-jetpack

1. use the version in the code below in installing NVidia packages
pip3 install torch==<correct_version>+nv22.02 torchvision==<correct_version>+nv22.02 --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v502

1. try this:
nvcc --version
nvidia-smi

1. check cuda added in local environment
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda

1. Check if Check if CUDA Toolkit is Installed
ls /usr/local/

1. Add CUDA to PATH
* Add CUDA binaries to your PATH:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

* Make the changes permanent by adding the lines above to your ~/.bashrc:
nano ~/.bashrc

add the lines save the file and reload
source ~/.bashrc

nvcc --version

Save and Exit:
Press Ctrl + O to save.
Press Ctrl + X to exit.




Pytorch for Jetson:
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

Jetson orin nano Cuda Cudnn torch torchauido torchvision:
https://forums.developer.nvidia.com/t/jetson-orin-nano-cuda-cudnn-torch-torchauido-torchvision/290430