import math
from xml.dom.pulldom import parseString
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from timm.models.layers.weight_init import trunc_normal_
from timm.models.vision_transformer import DropPath, Mlp


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, *args, **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, *args):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MlpBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x


class PatchMerge(nn.Module):
    def __init__(self, hidden_d, norm_layer):
        super().__init__()
        self.ln = norm_layer(hidden_d * 4)
        self.fc = nn.Linear(hidden_d * 4, hidden_d * 2, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        S = int(math.sqrt(N))
        x = x.transpose(1, 2).reshape(B, C, S, S)
        x = F.unfold(x, kernel_size=2, stride=2).transpose(1, 2)
        x = self.fc(self.ln(x))
        return x


class MlpPatchEmbed(nn.Module):
    def __init__(self, 
        img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
        stem_ratio=2., stem_depth=(1, 3, 0), act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        *args, **kwargs,
    ):
        super().__init__()
        k, d = patch_size // 4, embed_dim // 4
        stem_mlp = []
        for depth in stem_depth:
            if depth == 0:
                k, d = k * 2, d * 2
            else:
                if len(stem_mlp) == 0:
                    patchify_k, patchify_d = k, d
                stem_mlp.extend([
                    MlpBlock(
                        d, stem_ratio, act_layer=act_layer, norm_layer=norm_layer
                    ) for _ in range(depth)
                ])
                if d < embed_dim:
                    stem_mlp.extend([
                        PatchMerge(d, norm_layer),
                    ])
                    d = d * 2
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, patchify_d, kernel_size=patchify_k, stride=patchify_k)
        self.stem_mlp = nn.Sequential(*stem_mlp)

    def forward(self, x, *args):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.stem_mlp(x)
        return x


class ConvPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, *args, **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Sequential(*[
            nn.Conv2d(in_chans, embed_dim // 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 8, embed_dim // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        ])

    def forward(self, x, mask=None, *args):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if mask is not None:
            mask = mask.reshape(B, 1, *self.grid_size)
            mask = F.interpolate(mask, (H, W), mode='nearest')
            x = torch.where(mask > 0.5, torch.zeros_like(x), x)

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x
