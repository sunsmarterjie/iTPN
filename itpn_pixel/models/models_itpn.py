# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_
from timm.models.layers import to_2tuple


class Attention(nn.Module):
    def __init__(self, input_size, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * input_size - 1) * (2 * input_size - 1), num_heads)
        ) if rpe else None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpe_index=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if rpe_index is not None:
            S = int(math.sqrt(rpe_index.size(-1)))
            relative_position_bias = self.relative_position_bias_table[rpe_index].view(-1, S, S, self.num_heads)
            relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()
            attn = attn + relative_position_bias
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.float().clamp(min=torch.finfo(torch.float32).min, max=torch.finfo(torch.float32).max)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockWithRPE(nn.Module):
    def __init__(self, input_size, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., rpe=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        with_attn = num_heads > 0.

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = Attention(
            input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, rpe=rpe,
        ) if with_attn else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rpe_index=None, mask=None):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), rpe_index, mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B, -1,
            patches_resolution[0], self.inner_patches,
            patches_resolution[1], self.inner_patches,
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x):
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchSplit(nn.Module):
    def __init__(self, dim, fpn_dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.reduction = nn.Linear(dim, fpn_dim * 4, bias=False)
        self.fpn_dim = fpn_dim

    def forward(self, x):
        B, N, H, W, C = x.shape
        x = self.norm(x)
        x = self.reduction(x)
        x = x.reshape(
            B, N, H, W, 2, 2, self.fpn_dim
        ).permute(0, 1, 2, 4, 3, 5, 6).reshape(
            B, N, 2 * H, 2 * W, self.fpn_dim
        )
        return x


class iTPN(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=512, mlp_depth=3, depth=24,
                 fpn_dim=256, num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4., fpn_depth=2, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,  norm_layer=nn.LayerNorm, ape=True, rpe=True,
                 patch_norm=True, use_checkpoint=False, num_outs=1,
                 **kwargs):
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

        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, 2 * mlp_depth + depth))
        self.blocks = nn.ModuleList()

        self.blocks.extend([
            BlockWithRPE(
                Hp, mlvl_dims['4'], 0, bridge_mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer
            ) for _ in range(mlp_depth)]
        )
        self.blocks.append(PatchMerge(mlvl_dims['4'], norm_layer))
        self.blocks.extend([
            BlockWithRPE(
                Hp, mlvl_dims['8'], 0, bridge_mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer
            ) for _ in range(mlp_depth)]
        )
        self.blocks.append(PatchMerge(mlvl_dims['8'], norm_layer))
        self.blocks.extend([
            BlockWithRPE(
                Hp, mlvl_dims['16'], num_heads, mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer,
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

    def forward_features(self, x, ids_keep=None, mask=None):
        B, C, H, W = x.shape
        Hp, Wp = H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1]
        x = self.patch_embed(x)
        if ids_keep is not None:
            x = torch.gather(
                x, dim=1, index=ids_keep[:, :, None, None, None].expand(-1, -1, *x.shape[2:])
            )

        features = []
        for blk in self.blocks[:-self.num_main_blocks]:
            if isinstance(blk, PatchMerge):
                features.append(x)
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)

        x = x[..., 0, 0, :]
        if self.ape:
            pos_embed = self.interpolate_pos_encoding(x, Hp, Wp)
            if ids_keep is not None:
                pos_embed = torch.gather(
                    pos_embed.expand(B, -1, -1),
                    dim=1,
                    index=ids_keep[:, :, None].expand(-1, -1, pos_embed.shape[2]),
                )
            x += pos_embed
        x = self.pos_drop(x)

        rpe_index = None
        if self.rpe:
            if ids_keep is not None:
                B, L = ids_keep.shape
                rpe_index = self.relative_position_index
                rpe_index = torch.gather(
                    rpe_index[ids_keep, :], dim=-1, index=ids_keep[:, None, :].expand(-1, L, -1)
                ).reshape(B, -1)
            else:
                rpe_index = self.relative_position_index.view(-1)

        for blk in self.blocks[-self.num_main_blocks:]:
            x = checkpoint.checkpoint(blk, x, rpe_index, mask) if self.use_checkpoint else blk(x, rpe_index, mask)
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
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.fc_norm(x)
        x = self.head(x)
        return x


def itpn_base(**kwargs):
    model = iTPN(
        embed_dim=512, mlp_depth=3, depth=24, num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4.,
        rpe=True, num_outs=1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def itpn_large(**kwargs):
    model = iTPN(
        embed_dim=768, mlp_depth=2, depth=40, num_heads=12, bridge_mlp_ratio=3., mlp_ratio=4.,
        rpe=True, num_outs=1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

