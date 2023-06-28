# The code for object detection and instance segmentation with iTPN.

# get started

Please install [PyTorch](https://pytorch.org/). This codebase has been developed with python version 3.7, PyTorch version 1.8.0, CUDA 10.2 and torchvision 0.9.0. To get the full dependencies, please run:

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch

pip3 install -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html mmcv-full==1.4.0
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
    model.init_cfg['checkpoint']=$PRETRAINED \
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
