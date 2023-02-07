import ast
import os
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=8, help='the number of gpus')
parser.add_argument('--lr', type=float, default=4e-4, help='learning rate')
parser.add_argument('--pretrained', type=str, default='', help='the path of the pretrained chechpoints')
parser.add_argument('--configs', type=str, default='itpn/xxx.py', help='the path of the config file')
parser.add_argument('--rank', type=int, default=0, help='node rank')
parser.add_argument('--world_size', type=int, default=1, help='world size')
parser.add_argument('--samples_per_gpu', type=int, default=2, help='world size')

args, unparsed = parser.parse_known_args()

try:
    strs = 'python -m pip install --upgrade pip'
    os.system(strs)
    strs = 'pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex-master'
    os.system(strs)
except:
    print('Installing APEX failed!')

os.system('pip install mmcv-full==1.3.0 mmsegmentation==0.11.0')
os.system('pip install scipy timm==0.3.2')

cmd = f"bash tools/dist_train.sh \
        {args.configs} 8 \
        --work-dir ./output \
        --seed 0 \
        --deterministic \
        --options model.pretrained {args.pretrained} \
        data.samples_per_gpu={args.samples_per_gpu} \
        optimizer.lr={args.lr}"

os.system(cmd)
