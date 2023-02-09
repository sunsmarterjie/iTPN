import ast
import os
import argparse
import logging
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='the batch size per GPU')
parser.add_argument('--epochs', type=int, default=100, help='total fine-tuning epochs')
parser.add_argument('--warmup_epochs', type=int, default=5, help='the path of the config file')
parser.add_argument('--model', type=str, default='test_model', help='the path of the config file')
parser.add_argument('--dist_eval', action='store_true', default=True)
parser.add_argument('--weight', type=str, default='../weight.pth', help='the checkpoint file')
parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
parser.add_argument('--min_lr', type=float, default=5e-6, metavar='LR',
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--layer_decay', type=float, default=0.75,
                    help='layer-wise lr decay from ELECTRA/BEiT')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
parser.add_argument('--update_freq', default=1, type=int)
parser.add_argument('--clip_grad', type=float, default=5.0,
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--drop_path', type=float, default=.2,
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

parser.add_argument('--num_gpus', type=int, default=8, help='the number of gpus')
parser.add_argument('--rank', type=int, default=0, help='node rank')
parser.add_argument('--world_size', type=int, default=4, help='world size')
parser.add_argument('--data_path', default='/cache/imagenet/', type=str,
                    help='dataset path')
parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:6666')
args, unparsed = parser.parse_known_args()


master_addr = args.init_method[:-5]
master_port = args.init_method[-4:]
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port


cmd_str = f"python -m torch.distributed.launch \
    --nproc_per_node 4 \
    run_itpn_finetuning.py  \
    --data_path /home/TianYunjie/Workspace/dataset/ILSVRC2012_split_100/train \
    --eval_data_path /home/TianYunjie/Workspace/dataset/ILSVRC2012_split_100/val \
    --nb_classes {args.nb_classes} \
    --data_set 'image_folder' \
    --output_dir ../output \
    --input_size {args.input_size} \
    --log_dir ../output \
    --model {args.model} \
    --weight_decay {args.weight_decay}  \
    --batch_size {args.batch_size}  \
    --lr {args.blr} \
    --update_freq {args.update_freq}  \
    --warmup_epochs {args.warmup_epochs} \
    --epochs {args.epochs}  \
    --layer_decay {args.layer_decay} \
    --min_lr {args.min_lr} \
    --drop_path {args.drop_path}  \
    --mixup {args.mixup} \
    --cutmix {args.cutmix} \
    --imagenet_default_mean_and_std   \
    --dist_eval \
    --save_ckpt_freq 20 \
"

# cmd_str = f"python -m torch.distributed.launch \
#     --nproc_per_node 4 \
#     run_itpn_finetuning.py  \
#     --data_path /home/TianYunjie/Workspace/dataset/ILSVRC2012_split_100/train \
#     --eval_data_path /home/TianYunjie/Workspace/dataset/ILSVRC2012_split_100/val \
#     --nb_classes {args.nb_classes} \
#     --data_set 'image_folder' \
#     --output_dir ../output \
#     --input_size {args.input_size} \
#     --log_dir ../output \
#     --model {args.model} \
#     --weight_decay {args.weight_decay}  \
#     --finetune {args.weight}  \
#     --batch_size {args.batch_size}  \
#     --lr {args.blr} \
#     --update_freq {args.update_freq}  \
#     --warmup_epochs {args.warmup_epochs} \
#     --epochs {args.epochs}  \
#     --layer_decay {args.layer_decay} \
#     --min_lr {args.min_lr} \
#     --drop_path {args.drop_path}  \
#     --mixup {args.mixup} \
#     --cutmix {args.cutmix} \
#     --imagenet_default_mean_and_std   \
#     --dist_eval \
#     --save_ckpt_freq 20 \
# "

print('The running command is: ' + cmd_str)

os.system(cmd_str)
