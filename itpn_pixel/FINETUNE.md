


To fine-tune with **multi-node distributed training**, run the following on 4 nodes with 8 GPUs each.

***Script for iTPN-B:***

```bash
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 4 \
    --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=6666 \
    main_finetune.py \
    --world_size 4 \
    --batch_size 32 \
    --data_path ./path_to_data \
    --nb_classes 1000 \
    --output_dir  ../path_to_save \
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

***Script for iTPN-L:***

```bash
python -m torch.distributed.launch --nnodes 4 --nproc_per_node=8 --nnodes 4 
    --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=6666 \
    main_finetune.py \
    --world_size 4 \
    --batch_size 32 \
    --data_path ./path_to_data \
    --nb_classes 1000 \
    --output_dir  ../path_to_save \
    --model itpn_large \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 50 \
    --warmup_epochs 5 \
    --blr 1e-3  \
    --layer_decay 0.55 \
    --weight_decay 0.05 \
    --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --data_path ${IMAGENET_DIR}
```
