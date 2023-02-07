import ast
import os
import argparse
import logging
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--model', type=str, default='clip_tpn_base_3324_patch16_224', metavar='MODEL', 
                    help='the name of model to train')
parser.add_argument('--clip_path', type=str, default='../ViT-B-16.pt', 
                    help='the path of the CLIP model')
parser.add_argument('--input_size', default=224, type=int,
                    help='images input size for backbone')
parser.add_argument('--second_input_size', default=224, type=int,
                    help='images input size for CLIP teacher -- note that CLIP-L is 14x14 patch size')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')  

parser.add_argument('--num_mask_patches', default=75, type=int,
                        help='number of the visual tokens/patches need be masked')

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
parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:6666')
args, unparsed = parser.parse_known_args()


master_addr = args.init_method[:-5]
master_port = args.init_method[-4:]
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port

cmd_str = f"python -m torch.distributed.launch --nnodes={args.world_size} --nproc_per_node=8 \
            --node_rank={args.rank} --master_addr={master_addr} --master_port={master_port} \
            run_itpn_pretraining.py \
            --data_set=../imagenet/ \
            --data_path=../imagenet/train \
            --output_dir=../output \
            --log_dir=../output  \
            --model {args.model}  \
            --num_mask_patches {args.num_mask_patches}  \
            --second_input_size {args.second_input_size}  \
            --second_interpolation 'bicubic'  \
            --batch_size {args.batch_size} \
            --input_size {args.input_size} \
            --lr {args.blr}  \
            --warmup_epochs {args.warmup_epochs}  \
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
