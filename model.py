
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from einops import rearrange
import math
import numpy as np

from torch.nn import functional as F

import segmentation_models_pytorch as smp


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, d_v, n_head, temperature_scaling, split, dropout, d_qk, compute_v):
        super().__init__()
        self.d_v = d_v
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        self.w_qs = nn.Linear(embed_dim, n_head * d_qk, bias=False)
        self.w_ks = nn.Linear(embed_dim, n_head * d_qk, bias=False)
        self.w_vs = nn.Linear(d_v, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_v, bias=False)
        self.attention = None
        self.temperature = temperature_scaling
        self.d_k = d_qk
    
    def forward(self, q, k, v, qpos, kpos, qk_mask=None, k_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = v

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if qk_mask is not None:
            attn += qk_mask
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        # output += residual

        return output, attn


class PatchInpainting(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    def __init__(
        self,
        *,
        kernel_size: int,
        nheads: int,
        stem_out_stride: int = 1,
        stem_out_channels: int = 3,
        cross_attention: bool = False,
        mask_query_with_segmentation_mask: bool = False,
        initial_temperature_scaling: float = 1,
        final_temperature_scaling: float = 1,
        merge_mode: str = 'sum',
        use_kpos: bool = True,
        image_size: int = 512,
        embed_dim: int = 512,
        use_qpos: bool = True,
        dropout: float = 0.1,
        attention_type: str = 'ane_transformers.reference.multihead_attention.SelfAttention',
        compute_v: float = 0.1,
        feature_i: int = 3,
        feature_dim: int = 128,
        concat_features: bool = True,
        attention_masking: bool = True,
        final_conv: bool = False,
        mask_inpainting: bool = True,
        model,
    ):
        self.cross_attention = cross_attention
        self.kernel_size = kernel_size
        self.mask_query_with_segmentation_mask = mask_query_with_segmentation_mask
        self.nheads = nheads
        self.use_kpos = use_kpos
        self.use_qpos = use_qpos
        self.temperature_scaling = initial_temperature_scaling
        self.final_temperature_scaling = final_temperature_scaling
        self.feature_i = feature_i
        self.feature_dim = feature_dim
        print(f"PatchInpainting self.feature_dim: {self.feature_dim}")
        self.concat_features = concat_features
        self.attention_masking = attention_masking
        self.window_size= image_size // kernel_size
        self.final_conv = final_conv
        self.mask_inpainting = mask_inpainting
        super().__init__()
        self.pooling_layer = nn.MaxPool2d(
            kernel_size, stride=kernel_size)
        self.multihead_attention = MultiHeadAttention(
            embed_dim=stem_out_channels*kernel_size*kernel_size + self.feature_dim if self.concat_features else stem_out_channels*kernel_size*kernel_size, d_v=stem_out_channels*kernel_size*kernel_size, n_head=self.nheads,
            temperature_scaling=initial_temperature_scaling, split=True, dropout=dropout, d_qk=embed_dim, compute_v=compute_v
        )
        self.stem_out_channels = stem_out_channels
        self.stem_out_stride = stem_out_stride
        self.register_buffer('qk_mask', 1e4 * torch.eye(int((image_size /
                                                             stem_out_stride/self.kernel_size)**2)).unsqueeze(0).unsqueeze(0))
        if not mask_query_with_segmentation_mask:
            self.mask_query = torch.nn.Parameter(torch.zeros(
                1, int((image_size/stem_out_stride/self.kernel_size)**2), 1, 1).float())

        self.encoder_decoder = model
        self.image_size = image_size
        self.positionalencoding = torch.nn.Parameter(torch.zeros(1, self.kernel_size**2*stem_out_channels + self.feature_dim, int((image_size/stem_out_stride/self.kernel_size)**2))
                                                     ) if use_kpos or use_qpos else None
        self.final_conv = torch.nn.Sequential(nn.Conv2d(stem_out_channels*kernel_size*kernel_size, stem_out_channels*kernel_size*kernel_size, kernel_size=3, stride=1, padding=1, padding_mode='reflect'), torch.nn.Sigmoid()) if self.final_conv else None
        self.pixel_shuffle = nn.PixelShuffle(self.kernel_size)
        if merge_mode == 'all':
            self.merge_func = self.merge_all_patches_sum

        self.register_buffer(
            name="unfolding_weights",
            tensor=self._compute_unfolding_weights(),
            persistent=False,
        )
        self.register_buffer(
            name="unfolding_weights_image",
            tensor=self._compute_unfolding_weights_v2(),
            persistent=False,
        )
        self.register_buffer(
            name="unfolding_weights_mask",
            tensor=self._compute_unfolding_weights_v3(),
            persistent=False,
        )

    def _compute_unfolding_weights(self) -> torch.Tensor:
        weights = torch.eye(self.kernel_size *
                            self.kernel_size, dtype=torch.float)
        weights = weights.reshape(
            (self.kernel_size * self.kernel_size,
             1, self.kernel_size, self.kernel_size)
        )
        weights = weights.repeat(self.stem_out_channels, 1, 1, 1)
        return weights

    def _compute_unfolding_weights_v2(self) -> torch.Tensor:
        weights = torch.eye(self.kernel_size *
                            self.kernel_size, dtype=torch.float)
        weights = weights.reshape(
            (self.kernel_size * self.kernel_size,
             1, self.kernel_size, self.kernel_size)
        )
        weights = weights.repeat(3, 1, 1, 1)
        return weights

    def _compute_unfolding_weights_v3(self) -> torch.Tensor:
        weights = torch.eye(self.kernel_size *
                            self.kernel_size, dtype=torch.float)
        weights = weights.reshape(
            (self.kernel_size * self.kernel_size,
             1, self.kernel_size, self.kernel_size)
        )
        weights = weights.repeat(1, 1, 1, 1)
        return weights

    def unfolding_coreml(self, feature_map: torch.Tensor, weights):
        batch_size, in_channels, img_h, img_w = feature_map.shape
        patches = F.conv2d(
            feature_map,
            weights,
            bias=None,
            stride=(self.kernel_size, self.kernel_size),
            padding=0,
            dilation=1,
            groups=in_channels,
        )
        return patches, (img_h, img_w)

    def folding_coreml(self, patches: torch.Tensor, output_size) -> torch.Tensor:
        if self.final_conv:
            patches = rearrange(patches, 'b (p1 p2) c -> b c p1 p2', p1=self.window_size, p2=self.window_size)
            patches = self.final_conv(patches)
            patches = rearrange(patches, 'b c p1 p2 -> b (p1 p2) c')
        final_image = rearrange(patches, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                        h=self.image_size//self.kernel_size, w=self.image_size//self.kernel_size,
                        p1=self.kernel_size, p2=self.kernel_size)
        return final_image

    def forward(self, image, mask):
        image_coarse_inpainting, features = self.encoder_decoder(image)
        if self.mask_inpainting:
            image = image_coarse_inpainting*mask + image * (1 - mask)
        else:
            image = image_coarse_inpainting
        image_to_return = image_coarse_inpainting
        image_as_patches, sizes = self.unfolding_coreml(
            image, self.unfolding_weights)

        pos = self.positionalencoding.repeat(
            image_as_patches.size(0), 1, 1).unsqueeze(2) if self.use_qpos else None

        mask_same_res_as_features_pooled, _ = self.unfolding_coreml(
                mask, self.unfolding_weights_mask)
        mask_same_res_as_features_pooled = mask_same_res_as_features_pooled[:, 0:1, :, :]
        mask_same_res_as_features_pooled = mask_same_res_as_features_pooled.flatten(
                start_dim=2).unsqueeze(-1)

        if self.concat_features:
            features_to_concat = features[self.feature_i]
            features_to_concat = F.interpolate(features_to_concat, size=image_as_patches.shape[-2:], mode='bilinear', align_corners=False)
            input_attn = torch.cat([image_as_patches, features_to_concat],dim=1)
            input_attn = input_attn.flatten(start_dim=2).transpose(1, 2)
        else:
            input_attn = image_as_patches.flatten(start_dim=2).transpose(1, 2)

        image_as_patches = image_as_patches.flatten(start_dim=2).transpose(1, 2)
        
        qk_mask = -1e4*self.qk_mask.repeat(image_as_patches.size(0), 1, 1, 1) + 2e4*((1 - mask_same_res_as_features_pooled)*self.qk_mask) if self.attention_masking else None
        k_mask  = -1e4*mask_same_res_as_features_pooled if self.attention_masking else None
        out, atten_weights = self.multihead_attention(input_attn, input_attn,
            image_as_patches, qpos=pos, kpos=pos, qk_mask=qk_mask, k_mask=k_mask
        )

        out = self.folding_coreml(out, sizes)

        return out, atten_weights[0], image_to_return

    def merge_all_patches_sum(self, patch_scores, sequence_of_patches):
        return torch.einsum('bkhq,bchk->bchq', patch_scores, sequence_of_patches.unsqueeze(2)).squeeze(2)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, d_model, max_len)
        pe[0, 0::2, :] = torch.sin(position * div_term).T
        pe[0, 1::2, :] = torch.cos(position * div_term).T
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe[:, :, :x.size(-1)]
        return self.dropout(x)



class MobileOneCoarse(nn.Module):
    def __init__(self, encoder_name, encoder_weights, in_channels, classes):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
        
    def forward(self, x):
        features = self.model.encoder(x) # Restored line
        output_image = self.model.decoder(*features)
        output_image = self.model.segmentation_head(output_image)
        # Apply sigmoid to match the previous model's output activation
        output_image = torch.sigmoid(output_image)
        return output_image, features


class InpaintingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.coarse_model = MobileOneCoarse(**config['coarse_model']['parameters'])
        self.generator = PatchInpainting(**config['generator']['params'], model=self.coarse_model)

    def forward(self, image, mask):
        return self.generator(image, mask)


if __name__ == '__main__':
    # Dummy config based on train_default.conf, with model expecting 512x512 input
    config = {
        'coarse_model': {
            'class': 'MobileOneCoarse',
            'parameters': {
                'encoder_name': 'mobileone_s4',
                'encoder_weights': 'imagenet',
                'in_channels': 3,
                'classes': 3
            }
        },
        'generator': {
            'generator_class': 'PatchInpainting',
            'params': {
                'kernel_size': 8,
                'nheads': 1,
                'stem_out_stride': 1,
                'stem_out_channels': 3,
                'initial_temperature_scaling': 1,
                'final_temperature_scaling': 1,
                'merge_mode': 'all',
                'image_size': 512,
                'embed_dim': 576,
                'use_qpos': None,
                'use_kpos': None,
                'dropout': 0.1,
                'feature_i': 3,
                'concat_features': True,
                'final_conv': True,
                'feature_dim': 448,
                'attention_type': 'MultiHeadAttention',
                'compute_v': False
            }
        }
    }

    # Create a high-resolution dummy image and mask
    high_res_image = torch.randn(1, 3, 2048, 2048)
    high_res_mask = torch.zeros(1, 1, 2048, 2048)
    high_res_mask[:, :, 1024:, 1024:] = 1

    # Downsample inputs for the generator
    low_res_image = F.interpolate(high_res_image, size=512, mode='bicubic', antialias=True)
    low_res_mask = F.interpolate(high_res_mask, size=512)
    masked_low_res_image = low_res_image * (1 - low_res_mask)

    # Instantiate the model
    model = InpaintingModel(config)

    # Forward pass with the downsampled image
    output, attn_scores, temp_image = model(masked_low_res_image, low_res_mask)

    # Check that the output shape is the expected low resolution
    assert output.shape == (1, 3, 512, 512)
    print("Model test passed on downsampled image!")

