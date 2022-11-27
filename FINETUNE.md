## This file is being updated

## Fine-tuning Pre-trained iTPN for Classification

We will release the checkpoints [here](https://to_be_update).

To fine-tune with **multi-node distributed training**, run the following on 4 nodes with 8 GPUs each.

Script for iTPN-B:
```bash
python startup_ft.py \
    --world_size 4 \
    --batch_size 32 \
    --model itpn_base \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 4e-4  \
    --layer_decay 0.60 \
    --weight_decay 0.05 \
    --drop_path 0.15 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --data_path ${IMAGENET_DIR}
```

Script for iTPN-L:
```bash
python startup_ft.py \
    --world_size 4 \
    --batch_size 32 \
    --model itpn_large \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 50 \
    --warmup_epochs 5 \
    --blr 1e-3  \
    --layer_decay 0.55 \
    --weight_decay 0.05 \
    --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --data_path ${IMAGENET_DIR}
```

OR

you can run with 
Script for iTPN-B: 
```bash
python -m torch.distributed.launch --nnodes 4 --nproc_per_node=8 main_finetune.py \
    --world_size 4 \
    --batch_size 32 \
    --model itpn_large \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 4e-4  \
    --layer_decay 0.60 \
    --weight_decay 0.05 \
    --drop_path 0.15 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --data_path ${IMAGENET_DIR}
```

Script for iTPN-L: 
```bash
python -m torch.distributed.launch --nnodes 4 --nproc_per_node=8 main_finetune.py \
    --world_size 4 \
    --batch_size 32 \
    --model itpn_large \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 50 \
    --warmup_epochs 5 \
    --blr 1e-3  \
    --layer_decay 0.55 \
    --weight_decay 0.05 \
    --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --data_path ${IMAGENET_DIR}
```
