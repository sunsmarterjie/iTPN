

## Fast_iTPN Fine-Tuning on ImageNet-1K: 

<details>
 <summary> Fine-tune <b>Fast_iTPN-tiny</b>:</summary>

```bash    
NNODES=1
GPUS=2
CLASSES=100
INPUT_SIZE=224
WEIGHT_DECAY=0.05
BATCH_SIZE=32
LAYER_SCALE_INIT_VALUE=0.1
LR=5e-4
UPDATE_FREQ=1
REPROB=0.25
EPOCHS=50
W_EPOCHS=5
LAYER_DECAY=0.75
MIN_LR=1e-6
WARMUP_LR=1e-6
DROP_PATH=0.1
MIXUP=0.8
CUTMIX=1.0
SMOOTHING=0.1
MODEL='fast_itpn_large_2240_patch16_256'
WEIGHT='/home/TangXi/Workspace/fast_itpn_large_1600e_1k.pt'

cmd_str = f"python -m torch.distributed.launch \
    --nproc_per_node {GPUS} \
    --nnodes={NNODES} \
    run_class_finetuning.py  \
    --data_path /home/TangXi/Dataset/ILSVRC2012_split_100/train \
    --eval_data_path /home/TangXi/Dataset/ILSVRC2012_split_100/val \
    --nb_classes {CLASSES} \
    --data_set image_folder \
    --output_dir ./output \
    --input_size {INPUT_SIZE} \
    --log_dir ./output \
    --model {MODEL} \
    --weight_decay {WEIGHT_DECAY}  \
    --finetune {WEIGHT}  \
    --batch_size {BATCH_SIZE}  \
    --layer_scale_init_value {LAYER_SCALE_INIT_VALUE} \
    --lr {LR} \
    --update_freq {UPDATE_FREQ}  \
    --reprob {REPROB} \
    --warmup_epochs {W_EPOCHS} \
    --epochs {EPOCHS}  \
    --layer_decay {LAYER_DECAY} \
    --min_lr {MIN_LR} \
    --warmup_lr {WARMUP_LR} \
    --drop_path {DROP_PATH}  \
    --mixup {MIXUP} \
    --cutmix {CUTMIX} \
    --smoothing {SMOOTHING} \
    --imagenet_default_mean_and_std   \
    --dist_eval \
    --model_ema \
    --model_ema_eval \
    --save_ckpt_freq 20 \
"
```
</details>


<details>
 <summary> Fine-tune <b>iTPN-L/16</b>:</summary>


```bash    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666  run_itpn_finetuning.py \
    --world_size 4 \
    --batch_size 16 \
    --data_path ../path_to_data \
    --nb_classes 1000 \
    --output_dir  ../path_to_save  \
    --model itpn_large_2240_patch16_224 \
    --blr 2.0e-4 \
    --finetune ../path_to_checkpoint \
    --drop_path 0.25 \
    --epochs 50 \
    --input_size 224 \
    --layer_decay 0.55 \
    --update_freq 1 \
    --warmup_epochs 5 \
    --mixup 0.8 \
    --cutmix  1.0 \
    --weight_decay 0.05
```
</details>

