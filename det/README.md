# The code for object detection and instance segmentation with iTPN.

# get started

Please install [PyTorch](https://pytorch.org/). This codebase has been developed with python version 3.7, PyTorch version 1.8.0, CUDA 10.2 and torchvision 0.9.0. To get the full dependencies, please run:

```bash
conda create -n itpn_det python=3.7
source activate itpn_det

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch

pip install pyyaml==5.1 mmpycocotools==12.0.3 einops torchvision==0.9.0 cython==0.29.28 
pip install timm==0.5.4 pycocotools==2.0.4 numpy==1.21.5 terminaltables==3.1.10 six==1.16.0


# install mmcv-full
# get my whl from:
# baidu disk at https://pan.baidu.com/s/142qXu9tQMcynjd9AabqBeA?pwd=mmcv password:mmcv
# or google drive at https://drive.google.com/file/d/16HDPDWg81LIP-3Q5MBy3XC7afjhA7dU6/view?usp=sharing
# put the "mmcv-full" whl in the current directory
pip install ./mmcv_full-1.5.1-cp37-cp37m-manylinux1_x86_64.whl

# install apex
# you can download apex using "git clone https://github.com/NVIDIA/apex" or get my apex from:
# baidu disk at https://pan.baidu.com/s/1HoxIVfYLv0SrJ02iu_qTNA?pwd=apex password:apex
# or google drive at https://drive.google.com/file/d/16HDPDWg81LIP-3Q5MBy3XC7afjhA7dU6/view?usp=sharing
# put the "apex" folder in the current directory
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..


python setup.py develop

```

## Fine-tuning with Mask R-CNN
#### We use 32 V100 GPUs, $NNODES = 4.

- To train iTPN-B/16 with Mask R-CNN:
```bash
python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDRESS \
    --master_port=$PORT \
    ./tools/train.py  \
    ./configs/itpn/pixel_itpn_base_1x_ld090_dp005.py \
    --launcher pytorch \
    --work-dir $OUTPUT_DIR \
    --no-validate \
    --deterministic \
    --cfg-options model.backbone.use_checkpoint=True \
    model.init_cfg.checkpoint=$PRETRAINED \
```

- To evaluate Mask R-CNN:
```bash
python -m torch.distributed.launch --nproc_per_node=8 \
    ./tools/test.py \
    $CONFIG \
    $checkpoint # from pretrained above \
    --launcher pytorch \
    --eval bbox segm \
    --cfg-options model.backbone.use_checkpoint=True
```


### You can run other experiments by simply using the corresponding configs.
