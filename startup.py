import ast
import os
import argparse
import logging
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=1600, help='total pretraing epochs')
parser.add_argument('--warmup_epochs', type=int, default=40, help='warmup epochs')
parser.add_argument('--model', type=str, default='itpn_base_dec512d8b', help='the path of the config file')

parser.add_argument('--mask_ratio', default=0.75, type=float, help='ratio of the visual tokens/patches need be masked')
parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--num_gpus', type=int, default=8, help='the number of gpus')
parser.add_argument('--rank', type=int, default=0, help='node rank')
parser.add_argument('--world_size', type=int, default=8, help='world size')
args, unparsed = parser.parse_known_args()

master_host = os.environ['VC_WORKER_HOSTS'].split(',')[0]
master_addr = master_host.split(':')[0]
master_port = '8524'
# FLAGS.worldsize will be re-computed follow as FLAGS.ngpu*FLAGS.nodes_num
# FLAGS.rank will be re-computed in main_worker
modelarts_rank = args.rank  # ModelArts receive FLAGS.rank means node_rank
modelarts_world_size = args.world_size  # ModelArts receive FLAGS.worldsize means nodes_num
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port

print(f'IP: {master_addr},  Port: {master_port}')
print(f'modelarts rank {modelarts_rank}, world_size {modelarts_world_size}')

#######################################################################################################

cmd_str = f"python -m torch.distributed.launch --nproc_per_node {args.num_gpus} \
    --nnodes={modelarts_world_size} --node_rank={modelarts_rank} --master_addr={master_addr} \
    --master_port={master_port} main_pretrain.py --batch_size {args.batch_size} --epochs {args.epochs} \
    --model {args.model} --warmup_epochs {args.warmup_epochs} --mask_ratio {args.mask_ratio} \
    --blr {args.blr}"

print('The running command is: ' + cmd_str)

os.system(cmd_str)
