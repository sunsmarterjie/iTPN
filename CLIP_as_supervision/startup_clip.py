import ast
import os
import argparse
import logging
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='batch size per gpu')
parser.add_argument('--epochs', type=int, default=800, help='total pre-training epochs')
parser.add_argument('--warmup_epochs', type=int, default=10, help='the warmup epochs')
parser.add_argument('--model', type=str, default='clip_tpn_base_3324_patch16_224',
                    help='the path of the config file')
parser.add_argument('--clip_path', type=str, default='../ViT-B-16.pt', help='the path of the CLIP model')
parser.add_argument('--input_size', default=224, type=int,
                    help='images input size for backbone')
parser.add_argument('--second_input_size', default=224, type=int,
                    help='images input size for CLIP teacher -- note that CLIP-L is 14x14 patch size')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')  

parser.add_argument('--resume_path', default='', help='resume path of the checkpoint')

parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--beta', default=0.98, type=float, help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--blr', type=float, default=1.0e-3, metavar='LR',
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                    help="We use 0.1 for both base and large models -- which might not be the best setting")
parser.add_argument('--num_gpus', type=int, default=8, help='the number of gpus')
parser.add_argument('--rank', type=int, default=0, help='node rank')
parser.add_argument('--world_size', type=int, default=8, help='world size')

args, unparsed = parser.parse_known_args()

###########################################################################################################

master_host = os.environ['VC_WORKER_HOSTS'].split(',')[0]
master_addr = master_host.split(':')[0]
master_port = '8525'  # '8524'
modelarts_rank = args.rank  # ModelArts receive FLAGS.rank means node_rank
modelarts_world_size = args.world_size  # ModelArts receive FLAGS.worldsize means nodes_num
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port
#######################################################################################################
cmd_str = f"python -m torch.distributed.launch --nnodes={modelarts_world_size} --nproc_per_node=8 --node_rank={modelarts_rank} \
            --master_addr={master_addr} --master_port={master_port} run_iTPN_clip.py \
            --data_set=../imagenet/ \
            --data_path=../imagenet/train \
            --output_dir=../output \
            --log_dir=../output  \
            --model {args.model}  \
            --num_mask_patches 75  \
            --second_input_size {args.second_input_size}  \
            --second_interpolation 'bicubic'  \
            --batch_size {args.batch_size} \
            --input_size {args.input_size} \
            --lr {args.blr}  \
            --warmup_epochs 10  \
            --clip_grad 3.0  \
            --drop_path {args.drop_path}  \
            --layer_scale_init_value {args.layer_scale_init_value}  \
            --imagenet_default_mean_and_std  \
            --opt_betas 0.9 {args.beta}  \
            --opt_eps {args.opt_eps}   \
            --epochs {args.epochs} \
            --clip_path {args.clip_path} \
            --save_ckpt_freq 20 "

print('The running command is: ' + cmd_str)
os.system(cmd_str)
