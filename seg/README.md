
# The code for semantic segmentation (ADE20K) with iTPN.

## get started



```bash

conda create -n itpn_det python=3.7
source activate itpn_det

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch

pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
pip install scipy timm==0.3.2

# install apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

Follow [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) to prepare the ADE20k dataset.

## Finetune iTPN-B using UperNet:

```bash
bash tools/dist_train.sh \
    ./configs/itpn/pixel_upernet_itpn_base_12_512_slide_160k_ade20k_pt2ft.py 8 \
    --work-dir /path/to/save --seed 0  --deterministic \
    --options model.pretrained=<PRETRAIN_CHECKPOINT_PATH>
```


## Eval the fine-tuned checkpoints

```bash
bash tools/dist_test.sh \
    ./configs/itpn/pixel_upernet_itpn_base_12_512_slide_160k_ade20k_pt2ft.py \
    <FINETUNED_CHECKPOINT_PATH> 8 \
    --eval mIoU
```


### You can run other experiments by simply using the corresponding configs.
