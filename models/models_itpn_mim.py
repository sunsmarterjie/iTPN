
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------


from typing import Iterable
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import trunc_normal_
from .masked_autoencoder import MaskedAutoencoder
from .models_itpn import iTPN, PatchEmbed, PatchMerge, PatchSplit, BlockWithRPE
from util.pos_embed import get_2d_sincos_pos_embed


class iTPNMaskedAutoencoder(MaskedAutoencoder, iTPN):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, fpn_dim=256, fpn_depth=2,
                 embed_dim=512, mlp_depth=3, depth=24, num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=False,
                 num_outs=3, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 **kwargs):
        MaskedAutoencoder.__init__(self)
        assert num_outs in [1, 2, 3]
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
            self.align_dim_16tofpn = nn.Linear(embed_dim, fpn_dim) if embed_dim != fpn_dim else None
            self.fpn_modules = nn.ModuleList()
            self.fpn_modules.append(
                BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer
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
            self.fpn_modules.append(
                BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer,
                ))

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

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_patch_size = patch_size
        self.decoder_embed = nn.ModuleList()
        self.decoder_embed.append(nn.Sequential(
            norm_layer(fpn_dim),
            nn.Linear(fpn_dim, decoder_embed_dim, bias=True),
        ))

        if self.num_outs >= 2:
            self.decoder_embed.append(nn.Sequential(
                norm_layer(fpn_dim),
                nn.Linear(fpn_dim, decoder_embed_dim // 4, bias=True),
            ))
        if self.num_outs >= 3:
            self.decoder_embed.append(nn.Sequential(
                norm_layer(fpn_dim),
                nn.Linear(fpn_dim, decoder_embed_dim // 16, bias=True),
            ))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            BlockWithRPE(
                Hp, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias, qk_scale,
                rpe=False, norm_layer=norm_layer
            )
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.decoder_patch_size ** 2 * in_chans,
                                      bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.absolute_pos_embed.shape[-1], Hp, cls_token=False)
        self.absolute_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], Hp, cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def masking_id(self, batch_size, mask_ratio):
        N, L = batch_size, self.absolute_pos_embed.size(1)
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=self.absolute_pos_embed.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.absolute_pos_embed.device)
        mask[:, :ids_keep.size(1)] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def forward_encoder(self, x, mask_ratio):
        ids_keep, ids_restore, mask = self.masking_id(x.size(0), mask_ratio)

        x = self.forward_features(x, ids_keep=ids_keep)

        return x, mask, ids_restore

    def forward_decoder(self, inps, ids_restore):
        B, N, _, _, _ = inps[0].shape
        # embed tokens
        feats = []
        for feat, layer in zip(inps, self.decoder_embed):
            # x = layer(feat.reshape(B, N, -1))
            x = layer(feat).reshape(B, N, -1)
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            feats.append(x)
        x = feats.pop(0)
        # add pos embed
        x = x + self.decoder_pos_embed

        for i, feat in enumerate(feats):
            x = x + feats[i]
        # apply Transformer blocks
        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        return None, x


def itpn_base_dec512d8b(**kwargs):
    model = iTPNMaskedAutoencoder(
        embed_dim=512, mlp_depth=3, depth=24, num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4.,
        num_outs=3, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def itpn_large_dec512d8b(**kwargs):
    model = iTPNMaskedAutoencoder(
        embed_dim=768, mlp_depth=2, depth=40, num_heads=12, bridge_mlp_ratio=3., mlp_ratio=4.,
        num_outs=3, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


