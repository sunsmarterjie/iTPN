import clip
import torch.nn as nn
import torch


class clip_distill(nn.Module):
    def __init__(self, teacher_size=224, download_root='./pretrain/', model_name='ViT-B/16', **kwargs):
        super().__init__()
        self.scaling_layer = ScalingLayerForClip()
        self.teacher_model, _ = clip.load(model_name, device='cpu', jit=False, download_root=download_root)
        if 'L' in model_name:
            self.decoder_out_dim = 768
        else:
            self.decoder_out_dim = 512

        self.LN = nn.LayerNorm(self.decoder_out_dim, elementwise_affine=False)

        if self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad = False  # fix teacher_model model

            self.teacher_model.eval()
            self.teacher_input_size = teacher_size

    @torch.no_grad()
    def get_target(self, x, **kwargs):
        norm_imgs = self.scaling_layer(x)
        target = self.teacher_model.encode_image(norm_imgs, return_all_tokens=True) @ self.teacher_model.visual.proj
        return self.LN(target)

    def forward(self, x, **kwargs):
        """
        x: shape [B, 3, H, W] in [0, 1]
        """
        target = self.get_target(x, **kwargs)

        return target


class ScalingLayerForClip(nn.Module):
    def __init__(self):
        super(ScalingLayerForClip, self).__init__()
        self.register_buffer('shift', torch.Tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None])

    def forward(self, inp):
        inp = ((inp + 1.) * 127.5).clamp(0, 255.) / 255.  # rescale to [0, 1.]
        return (inp - self.shift) / self.scale
