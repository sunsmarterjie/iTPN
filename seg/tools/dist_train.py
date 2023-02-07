import os
import sys
import argparse
import yaml
import torch
import moxing as mox
mox.file.shift('os', 'mox')
import logging
import time

print('-------------')
os.system('pwd')
print('-------------')
os.system('ls')

mox.file.copy_parallel('s3://bucket-3690/tianyunjie/datasets/environment/torch-1.8.0+cu101-cp36-cp36m-linux_x86_64.whl','/cache/torch-1.8.0+cu101-cp36-cp36m-linux_x86_64.whl')
os.system('pip install /cache/torch-1.8.0+cu101-cp36-cp36m-linux_x86_64.whl')
mox.file.copy_parallel('s3://bucket-3690/tianyunjie/datasets/environment/torchvision-0.9.0+cu101-cp36-cp36m-linux_x86_64.whl','/cache/torchvision-0.9.0+cu101-cp36-cp36m-linux_x86_64.whl')
os.system('pip install /cache/torchvision-0.9.0+cu101-cp36-cp36m-linux_x86_64.whl')

try:
    strs = 'python -m pip install --upgrade pip'
    os.system(strs)
    strs = 'pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /home/ma-user/modelarts/user-job-dir/semantic_segmentation/apex-master'
    os.system(strs)
except:
    print('Installing APEX failed!')

# os.system('pip uninstall opencv-python')
os.system('cat /usr/local/cuda/version.txt')
os.system('nvcc --version')
os.system('pip install mmcv-full==1.3.0 mmsegmentation==0.11.0')
os.system('pip install scipy timm==0.3.2')
# os.system('pip install --ignore-installed opencv-python')
# os.system('pip uninstall opencv-python-headless')
# os.system('pip install opencv-python-headless==4.5.5.62')
# os.system('pip list |findstr opencv')

os.system('pip install --upgrade pip')
mox.file.copy_parallel('s3://bucket-3690/tianyunjie/datasets/ADE20k/ADEChallengeData2016.zip', '/cache/ADEChallengeData2016.zip')
os.system('unzip /cache/ADEChallengeData2016.zip -d /cache/')
print('ADEChallengeData2016 dirs: ' + str(list(os.listdir('/cache/'))))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(description=''):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--bucket', help='the bucket to copy dataset to local', type=str, default='bucket-3947')
    
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:6666')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--s3_path', type=str, default='s3://bucket-cv-competition-bj4/chenyabo/MAE_official/log/',
                    help='the path of the config file')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--data_url', dest='data_url', type=str, default='')
    parser.add_argument('--lr', type=float, default=4e-4, help='node rank')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--distributed', dest='distributed', help='which use distributed training', action='store_true', default=False)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument('--weight', type=str, default='/cache/weight.pth', help='the path of the config file')
    args, _ = parser.parse_known_args()
    return args

# run_root = os.path.join(os.path.abspath(os.path.dirname(__file__)))

run_root = '/home/ma-user/modelarts/user-job-dir/semantic_segmentation'
args = parse_args()


nodes_per_device = torch.cuda.device_count()
    
if args.pretrained is not None:
    mox.file.copy_parallel(args.pretrained, args.weight)
    logging.info('Done.')

    
if args.distributed:
    print('Using {} devices({} cards/each card) to train distributedly'.format(args.world_size, nodes_per_device))
    os.environ['nnodes'] = '%d'%args.world_size
    os.environ['node_rank'] = '%d'%args.rank
    args.init_method = os.environ['MA_VJ_NAME']+'-'+os.environ['MA_TASK_NAME']+'-0.'+os.environ['MA_VJ_NAME']+':6666'
    os.environ['master_addr'] = args.init_method[:-5]
    os.environ['master_port'] = args.init_method[-4:]
    print('master_addr: {}'.format(args.init_method[:-5]))
    print('master_addr: {}'.format(args.init_method[-4:]))
    
    
print('run_root: {}'.format(run_root))
cmd = 'cd {}'.format(run_root)
cmd += ' && sh tools/dist_train.sh configs/itpn/{} 8 --work-dir /cache/save --seed 0 --deterministic --options model.pretrained=/cache/weight.pth optimimzer.lr={}'
cmd = cmd.format(args.config, args.lr)
print(cmd)
os.system(cmd)


mox.file.copy_parallel("/cache/save", args.s3_path)