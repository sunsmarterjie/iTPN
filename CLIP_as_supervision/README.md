# [Integrally Pre-Trained Transformer Pyramid Networks](https://arxiv.org/abs/2106.08254)

Official PyTorch implementation and pretrained models of [iTPN](https://arxiv.org/pdf/2211.12735.pdf) (CLIP as supervision).

## iTPN Pre-Training on ImageNet-1K: 

<details>
 <summary> Pre-train <b>iTPN-B</b> using <b>CLIP-B</b>:</summary>

```bash    
python startup_clip.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_base_3324_patch16_224 \
    --beta 0.98 \
    --blr 1.5e-3 \
    --clip_path ../ViT-B-16.pt \
    --drop_path 0.1 \
    --epochs 300 \
    --input_size 224 \
    --layer_scale_init_value 0.1 \
    --opt_eps 1e-8 \
    --second_input_size 224 \
```
 
 OR 

```bash    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666  run_iTPN_clip.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_base_3324_patch16_224 \
    --beta 0.98 \
    --blr 1.5e-3 \
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
python startup_clip.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_base_3324_patch16_224 \
    --beta 0.98 \
    --blr 1.5e-3 \
    --clip_path ../ViT-L-14.pt \
    --drop_path 0.1 \
    --epochs 300 \
    --input_size 224 \
    --layer_scale_init_value 0.1 \
    --opt_eps 1e-8 \
    --second_input_size 196 \
```
 
 OR 

```bash    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666  run_iTPN_clip.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_base_3324_patch16_224 \
    --beta 0.98 \
    --blr 1.5e-3 \
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
python startup_clip.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_large_2240_patch16_224 \
    --beta 0.98 \
    --blr 1.5e-3 \
    --clip_path ../ViT-B-16.pt \
    --drop_path 0.2 \
    --epochs 300 \
    --input_size 224 \
    --layer_scale_init_value 0.1 \
    --opt_eps 1e-8 \
    --second_input_size 224 \
```
 
 OR 

```bash    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666  run_iTPN_clip.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_large_2240_patch16_224 \
    --beta 0.98 \
    --blr 1.5e-3 \
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
python startup_clip.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_large_2240_patch16_224 \
    --beta 0.98 \
    --blr 1.5e-3 \
    --clip_path ../ViT-L-14.pt \
    --drop_path 0.2 \
    --epochs 300 \
    --input_size 224 \
    --layer_scale_init_value 0.1 \
    --opt_eps 1e-8 \
    --second_input_size 196 \
```
 
 OR 

```bash    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666  run_iTPN_clip.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_large_2240_patch16_224 \
    --beta 0.98 \
    --blr 1.5e-3 \
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
python startup_clip.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_large_2240_patch16_256 \
    --beta 0.98 \
    --blr 1.5e-3 \
    --clip_path ../ViT-L-14.pt \
    --drop_path 0.2 \
    --epochs 300 \
    --input_size 256 \
    --layer_scale_init_value 0.1 \
    --opt_eps 1e-8 \
    --second_input_size 225 \
```
 
 OR 

```bash    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666  run_iTPN_clip.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_large_2240_patch16_256 \
    --beta 0.98 \
    --blr 1.5e-3 \
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

Fine-tune:
```bash
python startup_ft.py \
    --world_size 4 \
    --batch_size 16 \
    --model itpn_large_2240_patch16_224 \
    --blr 2.0e-4 \
    --pretrained ../path_to_pretrained \
    --drop_path 0.25 \
    --epochs 50 \
    --input_size 224 \
    --layer_decay 0.55 \
    --update_freq 2 \
    --warmup_epochs 5 \
```
