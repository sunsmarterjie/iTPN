## This file is being updated

## Pre-training iTPN using pixel supervision

To pre-train iTPN-B (recommended default) with **multi-node distributed training**, run the following on 8 nodes with 8 GPUs each:
```
python startup.py \
    --nodes 8 \
    --batch_size 64 \
    --model itpn_base_dec512d8b \
    --mask_ratio 0.75 \
    --epochs 1600 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}
```
or 
python -m torch.distributed.launch --nnodes 8 --nproc_per_node=8 main_pretrain.py \
        --batch_size 64
        --model itpn_base_dec512d8b \
        --blr 1.5e-4 \
        --mask_ratio 0.75
        --warmup_epochs 40 \
        --epochs 1600 \
        --data_path IN1K_path \ 
```

To train iTPN-L or iTPN-H (for future), set `--model itpn_large_dec512d8b` or `--model itpn_huge_dec512d8b`.
