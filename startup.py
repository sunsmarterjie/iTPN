import ast
import os
import argparse
import logging
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=1600, help='total pretraing epochs')
parser.add_argument('--warmup_epochs', type=int, default=40, help='warmup epochs')
parser.add_argument('--model', type=str, default='itpn_base_dec512d8b', help='the default model')
parser.add_argument('--data_path', type=str, default='../imagenet', help='the path of the dataset')
parser.add_argument('--mask_ratio', default=0.75, type=float, help='ratio of the visual tokens/patches need be masked')
parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

parser.add_argument('--num_gpus', type=int, default=8, help='the number of gpus')
parser.add_argument('--rank', type=int, default=0, help='get your machine node rank')
parser.add_argument('--world_size', type=int, default=8, help='how many machine you use')
parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:6666')
args, unparsed = parser.parse_known_args()

master_addr = args.init_method[:-5]
master_port = args.init_method[-4:]
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port

cmd_str = f"python -m torch.distributed.launch --nproc_per_node {args.num_gpus} \
    --nnodes={args.world_size} --master_addr={master_addr} --master_port={master_port} \
    main_pretrain.py --batch_size {args.batch_size} --epochs {args.epochs} \
    --model {args.model} --warmup_epochs {args.warmup_epochs} --mask_ratio {args.mask_ratio} \
    --blr {args.blr} --data_path {args.data_path}"

print('The running command is: ' + cmd_str)

os.system(cmd_str)
