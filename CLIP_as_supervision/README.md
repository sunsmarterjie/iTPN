# [Integrally Pre-Trained Transformer Pyramid Networks](https://arxiv.org/abs/2106.08254)

Official PyTorch implementation and pretrained models of [iTPN](https://arxiv.org/pdf/2211.12735.pdf) (CLIP as supervision).

***Scripts of pre-training iTPN-B using CLIP-B:***

```bash    
python startup_clip.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_base_3324_patch16_224 \
    --beta 0.98 \
    --blr 1.5e-3 \
    --clip_path ../path_to_clip_B \
    --drop_path 0.1 \
    --epochs 300 \
    --input_size 224 \
    --layer_scale_init_value 0.1 \
    --opt_eps 1e-8 \
    --second_input_size 224 \
```

***Scripts of pre-training iTPN-B using CLIP-L:***
```bash
python startup_clip.py \
    --world_size 8 \
    --batch_size 32 \
    --model clip_tpn_large_2240_patch16_224 \
    --beta 0.98 \
    --blr 1.5e-3 \
    --clip_path ../path_to_clip_L \
    --drop_path 0.2 \
    --epochs 300 \
    --input_size 224 \
    --layer_scale_init_value 0.1 \
    --opt_eps 1e-8 \
    --second_input_size 196 \
```

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
