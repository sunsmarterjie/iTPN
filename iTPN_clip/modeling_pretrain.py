# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# By Yunjie Tian
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# --------------------------------------------------------'
import torch
import torch.nn as nn
from functools import partial
import math

from modeling_finetune import _cfg
from timm.models.registry import register_model
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_
import torch.utils.checkpoint as checkpoint

from modeling_finetune import PatchMerge, PatchSplit, BlockWithRPE, PatchEmbed


class iTPNClipDistill(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, fpn_dim=256, fpn_depth=2,
                 embed_dim=512, mlp_depth1=3, mlp_depth2=3, depth=24, num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, init_values=None,
                 norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=False, teacher_dim=512,
                 num_outs=1, init_std=0.02, **kwargs):
        super().__init__()
        assert num_outs in [1, 2, 3, 4, 5]
        self.num_classes = num_classes
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_outs = num_outs
        self.num_main_blocks = depth
        self.fpn_dim = fpn_dim
        self.mlp_depth1 = mlp_depth1
        self.mlp_depth2 = mlp_depth2
        self.depth = depth

        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        Hp, Wp = self.patch_embed.patches_resolution
        assert Hp == Wp

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.num_features)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += Hp - 1
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, mlp_depth1 + mlp_depth2 + depth))
        dpr_cls = list(x.item() for x in torch.linspace(0, drop_path_rate, mlp_depth1 + mlp_depth2 + depth))
        self.blocks = nn.ModuleList()

        self.blocks.extend([
            BlockWithRPE(
                Hp, mlvl_dims['4'], 0, bridge_mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer, init_values=init_values
            ) for _ in range(mlp_depth1)]
        )
        self.blocks.append(PatchMerge(mlvl_dims['4'], norm_layer))
        self.blocks.extend([
            BlockWithRPE(
                Hp, mlvl_dims['8'], 0, bridge_mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer, init_values=init_values
            ) for _ in range(mlp_depth2)]
        )
        self.blocks.append(PatchMerge(mlvl_dims['8'], norm_layer))
        self.blocks.extend([
            BlockWithRPE(
                Hp, mlvl_dims['16'], num_heads, mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer, init_values=init_values
            ) for _ in range(depth)]
        )

        ########################### FPN PART ###########################
        if self.num_outs > 1:
            if embed_dim != fpn_dim:
                self.align_dim_16tofpn = nn.Linear(embed_dim, fpn_dim)
            else:
                self.align_dim_16tofpn = None
            self.fpn_modules = nn.ModuleList()
            self.fpn_modules.append(
                BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer
                ))
            self.fpn_modules.append(
                BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer,
                ))

            self.align_dim_16to8 = nn.Linear(mlvl_dims['8'], fpn_dim, bias=False)
            self.split_16to8 = PatchSplit(mlvl_dims['16'], fpn_dim, norm_layer)
            self.block_16to8 = nn.Sequential(
                *[BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer,
                ) for _ in range(fpn_depth)]
            )

        if self.num_outs > 2:
            self.align_dim_8to4 = nn.Linear(mlvl_dims['4'], fpn_dim, bias=False)
            self.split_8to4 = PatchSplit(fpn_dim, fpn_dim, norm_layer)
            self.block_8to4 = nn.Sequential(
                *[BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer,
                ) for _ in range(fpn_depth)]
            )
            self.fpn_modules.append(
                BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer
                )
            )

        if self.num_outs == 1:
            self.fc_norm = norm_layer(self.num_features)
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        ## merge the output
        self.decoder_embed = nn.ModuleList()
        self.decoder_embed.append(nn.Sequential(
            norm_layer(fpn_dim),
            nn.Linear(fpn_dim, embed_dim, bias=True),
        ))

        if self.num_outs >= 2:
            self.decoder_embed.append(nn.Sequential(
                norm_layer(fpn_dim),
                nn.Linear(fpn_dim, embed_dim // 4, bias=True),
            ))
        if self.num_outs >= 3:
            self.decoder_embed.append(nn.Sequential(
                norm_layer(fpn_dim),
                nn.Linear(fpn_dim, embed_dim // 16, bias=True),
            ))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.lm_head = nn.Linear(embed_dim, teacher_dim)
        self.lm_cls_head = nn.Linear(embed_dim, teacher_dim)
        self.norm = norm_layer(embed_dim)
        self.norm_cls = norm_layer(embed_dim)
        trunc_normal_(self.mask_token, std=init_std)
        trunc_normal_(self.lm_head.weight, std=init_std)
        trunc_normal_(self.lm_cls_head.weight, std=init_std)

        self.cls_pt_layers = nn.ModuleList(
            [
                BlockWithRPE(input_size=Hp, dim=mlvl_dims['16'], num_heads=num_heads, mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, drop_path=dpr_cls[self.mlp_depth1 + self.mlp_depth2 + i],
                             init_values=init_values) for i in range(int(self.depth * 0.75), int(self.depth * 0.75) + 2)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1]
        N = self.absolute_pos_embed.shape[1]
        if npatch == N and w == h:
            return self.absolute_pos_embed
        patch_pos_embed = self.absolute_pos_embed
        dim = x.shape[-1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w + 0.1, h + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward_features_for_pretraining(self, x, bool_masked_pos):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # B*L*4*4*C
        Hp, Wp = H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1]

        features = []
        for blk in self.blocks[:-self.num_main_blocks]:
            if isinstance(blk, PatchMerge):
                features.append(x)
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)

        x = x[..., 0, 0, :]
        batch_size, seq_len, _ = x.size()
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        if self.ape:
            pos_embed = self.interpolate_pos_encoding(x, Hp, Wp)
            x += pos_embed
        x = self.pos_drop(x)

        rpe_index = None
        if self.rpe:
            rpe_index = self.relative_position_index.view(-1)

        for i, blk in enumerate(self.blocks[-self.num_main_blocks:]):
            x = checkpoint.checkpoint(blk, x, rpe_index) if self.use_checkpoint else blk(x, rpe_index)
            if i == int(0.75 * self.depth):
                aux_out = x
        if self.num_outs == 1:
            return x

        ##########################  FPN forward  ########################

        x = x[..., None, None, :]
        outs = [x] if self.align_dim_16tofpn is None else [self.align_dim_16tofpn(x)]
        if self.num_outs >= 2:
            x = self.block_16to8(self.split_16to8(x) + self.align_dim_16to8(features[1]))
            outs.append(x)
        if self.num_outs >= 3:
            x = self.block_8to4(self.split_8to4(x) + self.align_dim_8to4(features[0]))
            outs.append(x)
        if rpe_index is None and self.num_outs > 3:
            outs = [
                out.reshape(B, Hp, Wp, *out.shape[-3:]).permute(0, 5, 1, 3, 2, 4).reshape(
                    B, -1, Hp * out.shape[-3], Wp * out.shape[-2]).contiguous()
                for out in outs]
            if self.num_outs >= 4:
                outs.insert(0, F.avg_pool2d(outs[0], kernel_size=2, stride=2))
            if self.num_outs >= 5:
                outs.insert(0, F.avg_pool2d(outs[0], kernel_size=2, stride=2))

        for i, out in enumerate(outs):
            out = self.fpn_modules[i](out)
            outs[i] = out

        for blk in self.cls_pt_layers:
            aux_out = blk(aux_out, rpe_index)
        return outs, aux_out

    def forward(self, x, bool_masked_pos):
        outs, aux_out = self.forward_features_for_pretraining(x, bool_masked_pos=bool_masked_pos)

        B, N, _ = aux_out.shape
        feats = []
        for feat, layer in zip(outs, self.decoder_embed):
            x = layer(feat).reshape(B, N, -1)
            feats.append(x)
        x = feats.pop(0)
        for i, feat in enumerate(feats):
            x = x + feats[i]

        # return the masked tokens
        x = self.norm(x)
        aux_x = self.norm_cls(aux_out)
        # return [self.lm_head(x[bool_masked_pos]), self.lm_cls_head(aux_x[bool_masked_pos])]
        return [self.lm_head(x), self.lm_cls_head(aux_x)]  # we supervise all tokens as default


@register_model
def clip_tpn_base_3324_patch16_224(pretrained=False, **kwargs):
    model = iTPNClipDistill(
        patch_size=16, embed_dim=512, mlp_depth1=3, mlp_depth2=3, depth=24, num_heads=8, bridge_mlp_ratio=3.,
        mlp_ratio=4, num_outs=3, fpn_dim=256, fpn_depth=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def clip_tpn_large_2240_patch16_224(pretrained=False, **kwargs):
    model = iTPNClipDistill(
        patch_size=16, embed_dim=768, mlp_depth1=2, mlp_depth2=2, depth=40, num_heads=12, bridge_mlp_ratio=3.,
        mlp_ratio=4, num_outs=3, fpn_dim=256, fpn_depth=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def clip_tpn_large_2240_patch16_256(pretrained=False, **kwargs):
    model = iTPNClipDistill(
        img_size=256,
        patch_size=16, embed_dim=768, mlp_depth1=2, mlp_depth2=2, depth=40, num_heads=12,
        bridge_mlp_ratio=3., mlp_ratio=4, num_outs=3, fpn_dim=256, fpn_depth=1, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
