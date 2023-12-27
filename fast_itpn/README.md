

## Fast_iTPN Fine-Tuning on ImageNet-1K: 

<details>
 <summary> Fine-tune <b>Fast_iTPN-tiny</b>:</summary>

```bash    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666  run_itpn_finetuning.py \
    --world_size 4 \
    --batch_size 32 \
    --data_path ../path_to_data \
    --nb_classes 1000 \
    --output_dir  ../path_to_save  \
    --model itpn_base_3324_patch16_224 \
    --blr 5.0e-4 \
    --finetune ../path_to_checkpoint \
    --drop_path 0.1 \
    --epochs 100 \
    --input_size 224 \
    --layer_decay 0.60 \
    --update_freq 1 \
    --warmup_epochs 20 \
    --mixup 0.8 \
    --cutmix  1.0 \
    --weight_decay 0.05
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

