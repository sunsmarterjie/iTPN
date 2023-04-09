# The code for object detection and instance segmentation with iTPN.

# get started

Please install [PyTorch](https://pytorch.org/). This codebase has been developed with python version 3.7, PyTorch version 1.8.0, CUDA 10.2 and torchvision 0.9.0. To get the full dependencies, please run:

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch

pip3 install -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html mmcv-full==1.5.1
pip3 install pytest-runner scipy tensorboardX faiss-gpu==1.6.1 tqdm lmdb sklearn pyarrow==2.0.0 timm DALL-E munkres six einops

# install apex
pip3 install git+https://github.com/NVIDIA/apex \
    --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"

# install mmdetection for object detection & instance segmentation
git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
cd Swin-Transformer-Object-Detection
pip3 install -r requirements/build.txt
pip3 install -v -e .
cd ..
```
