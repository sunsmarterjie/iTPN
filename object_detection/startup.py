import ast
import os
import argparse
import torch
import numpy as np
from collections import OrderedDict


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str, default='./pretrain.pth', help='the path of the ckp')
parser.add_argument('--dataset', type=str, default='coco2017', help='the path of the ckp')
parser.add_argument('--config', type=str, default='configs/itpn/...', help='the path of the ckp')
parser.add_argument('--rank', type=int, default=0, help='node rank')
parser.add_argument('--lr', type=float, default=0.0003, help='world size')
parser.add_argument('--drop_path_rate', type=float, default=0.2, help='world size')
parser.add_argument('--world_size', type=int, default=2, help='world size')
parser.add_argument('--samples_per_gpu', type=int, default=2, help='node rank')
parser.add_argument('--update_interval', type=int, default=2, help='node rank')
parser.add_argument('--pre_pip', type=ast.literal_eval, default='False', help='whether pip install in advance')
args, unparsed = parser.parse_known_args()

os.system(
    'pip install pyyaml==5.1 mmpycocotools==12.0.3 einops torchvision==0.9.0 cython==0.29.28 numpy==1.21.5 numpy-base==1.21.5 terminaltables==3.1.10 six==1.16.0')
os.system('pip install timm==0.5.4 pycocotools==2.0.4 einops')
os.system('pip install ./mmcv_full-1.5.1-cp37-cp37m-manylinux1_x86_64.whl')
os.system(
    'pip install -q --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex/')


cmd_str = f"python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes={modelarts_world_size} ./tools/train.py {args.config} --cfg-options \
    data.train.data_root=../datasets/coco \
    data.val.data_root=../datasets/coco \
    data.test.data_root=../datasets/coco \
    model.backbone.use_checkpoint=True \
    data.samples_per_gpu={args.samples_per_gpu} \
    optimizer.lr={args.lr} \
    --work-dir=./output \
    --no-validate \
    --gpus=8 \
    --deterministic \
    --launcher pytorch"

print('The running command is: ' + cmd_str)

os.system(cmd_str)
