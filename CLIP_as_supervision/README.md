# [Integrally Pre-Trained Transformer Pyramid Networks](https://arxiv.org/abs/2106.08254)

Official PyTorch implementation and pretrained models of iTPN (CLIP as supervision).

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
