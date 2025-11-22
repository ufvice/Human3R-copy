# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
from torch import nn
import math
import torch.nn.functional as F

# [FIX] Define a bilinear-only interpolation function for DINOv2
def interpolate_pos_encoding_bilinear(self, x, w, h):
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return self.pos_embed
    
    pos_embed = self.pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patch_size
    h0 = h // self.patch_size
    
    # We add a small number to see if N is close to square
    # (This logic is copied from original DINOv2 but adapted for bilinear)
    M = int(math.sqrt(N))
    assert N == M * M
    
    kwargs = {}
    
    # Force bilinear interpolation
    kwargs["mode"] = "bilinear"
    kwargs["align_corners"] = False
        
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
        size=(h0, w0),
        **kwargs,
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)


class Dinov2Backbone(nn.Module):
    def __init__(self, name='dinov2_vitb14', pretrained=False, *args, **kwargs):
        super().__init__()
        self.name = name
        # Load the model from hub
        self.encoder = torch.hub.load('facebookresearch/dinov2', self.name, pretrained=pretrained)
        
        # [FIX] Monkey patch the encoder's interpolate_pos_encoding method
        # Bind the new method to the encoder instance
        self.encoder.interpolate_pos_encoding = interpolate_pos_encoding_bilinear.__get__(self.encoder, self.encoder.__class__)

        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.encoder.embed_dim

    def forward(self, x):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert len(x.shape) == 4
        # The encoder will now use our bilinear interpolation when calling prepare_tokens -> interpolate_pos_encoding
        y = self.encoder.get_intermediate_layers(x)[0] # ViT-L+896x896: [bs,4096,1024] - [bs,nb_patches,emb]
        return y
