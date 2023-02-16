# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import ast
import os
import argparse
import logging
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='the batch size per GPU')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--model', type=str, default='itpn_base', metavar='MODEL',
                    help='the name of model to train')
parser.add_argument('--dist_eval', action='store_true', default=True, 
                   help='enabling distributed evaluation')
parser.add_argument('--weight', type=str, default='../weight.pth', help='the path of the checkpoint file')
parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--layer_decay', type=float, default=0.65,
                    help='layer-wise lr decay from ELECTRA/BEiT')
parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--clip_grad', type=float, default=5.0,
                    help='Clip gradient norm (default: 5.0)')
parser.add_argument('--drop_path', type=float, default=.1,
                    help='Drop path rate (default: 0.1)')

parser.add_argument('--num_gpus', type=int, default=8, help='the number of gpus')
parser.add_argument('--rank', type=int, default=0, help='node rank')
parser.add_argument('--world_size', type=int, default=4, help='world size')
parser.add_argument('--data_path', default='../imagenet/', type=str,
                    help='dataset path')
parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:6666')
args, unparsed = parser.parse_known_args()

master_addr = MASTER_ADDR  # get your master_addr
master_port = MASTER_PORT  # get your master_port
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port

cmd_str = f"python -m torch.distributed.launch --nproc_per_node {args.num_gpus} \
    --nnodes={args.wprld_size} --node_rank={args.rank} --master_addr={master_addr} \
    --master_port={master_port} main_finetune.py --finetune {args.weight} --batch_size {args.batch_size} \
    --epochs {args.epochs} --model {args.model} --warmup_epochs {args.warmup_epochs} \
    --clip_grad {args.clip_grad} --blr {args.blr} --drop_path {args.drop_path}  --min_lr {args.min_lr} \
    --layer_decay {args.layer_decay} --mixup {args.mixup} --cutmix {args.cutmix}"

print('The running command is: ' + cmd_str)

os.system(cmd_str)
