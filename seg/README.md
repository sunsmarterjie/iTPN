
# The code for semantic segmentation (ADE20K) with iTPN.

## get started



```bash
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
pip install scipy timm==0.3.2

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Follow [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) to prepare the ADE20k dataset.

## Finetune iTPN-B using UperNet:

```bash
bash tools/dist_train.sh \
    ./configs/itpn/pixel_upernet_itpn_base_12_512_slide_160k_ade20k_pt2ft.py 8 \
    --work-dir /path/to/save --seed 0  --deterministic \
    --options model.pretrained=<PRETRAIN_CHECKPOINT_PATH>
```
