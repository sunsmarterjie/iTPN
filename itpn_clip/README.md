# [Integrally Pre-Trained Transformer Pyramid Networks](https://arxiv.org/abs/2106.08254)

Official PyTorch implementation and pretrained models of [iTPN](https://arxiv.org/pdf/2211.12735.pdf) (CLIP as supervision).

## iTPN Pre-Training on ImageNet-1K: 

<details>
 <summary> Pre-train <b>iTPN-B</b> using <b>CLIP-B</b>:</summary>

```bash    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666  run_itpn_pretraining.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_base_3324_patch16_224 \
    --opt_betas 0.98 \
    --lr 1.5e-3 \
    --clip_path ../ViT-B-16.pt \
    --drop_path 0.1 \
    --epochs 300 \
    --input_size 224 \
    --layer_scale_init_value 0.1 \
    --opt_eps 1e-8 \
    --second_input_size 224 \
```
</details>



<details>
 <summary> Pre-train <b>iTPN-B</b> using <b>CLIP-L</b>:</summary>


```bash    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666  run_itpn_pretraining.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_base_3324_patch16_224 \
    --opt_betas 0.98 \
    --lr 1.5e-3 \
    --clip_path ../ViT-L-14.pt \
    --drop_path 0.1 \
    --epochs 300 \
    --input_size 224 \
    --layer_scale_init_value 0.1 \
    --opt_eps 1e-8 \
    --second_input_size 196 \
```
</details>



<details>
 <summary> Pre-train <b>iTPN-L/16</b> using <b>CLIP-B</b>:</summary>


```bash    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666  run_itpn_pretraining.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_large_2240_patch16_224 \
    --opt_betas 0.98 \
    --lr 1.5e-3 \
    --clip_path ../ViT-B-16.pt \
    --drop_path 0.2 \
    --epochs 300 \
    --input_size 224 \
    --layer_scale_init_value 0.1 \
    --opt_eps 1e-8 \
    --second_input_size 224 \
```
</details>


<details>
 <summary> Pre-train <b>iTPN-L/16</b> using <b>CLIP-L</b>:</summary>

```bash    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666  run_itpn_pretraining.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_large_2240_patch16_224 \
    --opt_betas 0.98 \
    --lr 1.5e-3 \
    --clip_path ../ViT-L-14.pt \
    --drop_path 0.2 \
    --epochs 300 \
    --input_size 224 \
    --layer_scale_init_value 0.1 \
    --opt_eps 1e-8 \
    --second_input_size 196 \
```
</details>


<details>
 <summary> Pre-train <b>iTPN-L/14</b> using <b>CLIP-L</b>:</summary>


```bash    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666  run_itpn_pretraining.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_large_2240_patch16_256 \
    --opt_betas 0.98 \
    --lr 1.5e-3 \
    --clip_path ../ViT-L-14.pt \
    --drop_path 0.2 \
    --epochs 300 \
    --input_size 256 \
    --layer_scale_init_value 0.1 \
    --opt_eps 1e-8 \
    --second_input_size 224 \
```
</details>


## iTPN Fine-Tuning on ImageNet-1K: 

<details>
 <summary> Fine-tune <b>iTPN-B</b>:</summary>

```bash    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666  run_itpn_finetuning.py \
    --world_size 4 \
    --batch_size 32 \
    --data_path ../path_to_data \
    --nb_classes 1000 \
    --output_dir  ../path_to_save  \
    --model itpn_base_3324_patch16_224 \
    --lr 5.0e-4 \
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
    --lr 2.0e-4 \
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

<details>
 <summary> Fine-tune <b>iTPN-L/14</b>:</summary>


```bash    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666  run_itpn_finetuning.py \
    --world_size 4 \
    --batch_size 16 \
    --data_path ../path_to_data \
    --nb_classes 1000 \
    --output_dir  ../path_to_save  \
    --model itpn_large_2240_patch16_256 \
    --lr 2.0e-4 \
    --finetune ../path_to_checkpoint \
    --drop_path 0.25 \
    --epochs 50 \
    --input_size 256 \
    --layer_decay 0.55 \
    --update_freq 1 \
    --warmup_epochs 5 \
    --mixup 0.8 \
    --cutmix  1.0 \
    --weight_decay 0.05
```
</details>


