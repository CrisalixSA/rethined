
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from einops import rearrange
import math
import numpy as np

from torch.nn import functional as F

from kornia.filters import GaussianBlur2d

from mobileone import MobileOne, mobileone


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, d_v, n_head, split, dropout, d_qk, compute_v, use_argmax=False):
        super().__init__()
        self.d_v = d_v
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        self.w_qs = nn.Linear(embed_dim, n_head * d_qk, bias=False)
        self.w_ks = nn.Linear(embed_dim, n_head * d_qk, bias=False)
        self.w_vs = nn.Linear(d_v, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_v, bias=False)
        self.attention = None
        self.d_k = d_qk
        self.use_argmax = use_argmax
    
    def forward(self, q, k, v, qpos, kpos, qk_mask=None, k_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = v

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        attn = torch.matmul(q / self.d_k**0.5, k.transpose(2, 3))

        if qk_mask is not None:
            attn += qk_mask
        
        attn = F.softmax(attn, dim=-1)

        if self.use_argmax:
            idx =  torch.argmax(attn, dim=1, keepdims=True)
            attn = torch.zeros_like(attn).scatter_(1, idx, 1.)

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
        use_argmax: bool = False,
        model,
    ):
        self.cross_attention = cross_attention
        self.kernel_size = kernel_size
        self.mask_query_with_segmentation_mask = mask_query_with_segmentation_mask
        self.nheads = nheads
        self.use_kpos = use_kpos
        self.use_qpos = use_qpos
        self.feature_i = feature_i
        self.feature_dim = feature_dim
        self.concat_features = concat_features
        self.attention_masking = attention_masking
        self.window_size= image_size // kernel_size
        self.final_conv = final_conv
        self.mask_inpainting = mask_inpainting
        self.use_argmax = use_argmax
        super().__init__()
        self.final_gaussian_blur = GaussianBlur2d((7,7),sigma=(2.01, 2.01),separable=False)
        self.pooling_layer = nn.MaxPool2d(
            kernel_size, stride=kernel_size)
        self.multihead_attention = MultiHeadAttention(
            embed_dim=stem_out_channels*kernel_size*kernel_size + self.feature_dim if self.concat_features else stem_out_channels*kernel_size*kernel_size, d_v=stem_out_channels*kernel_size*kernel_size, n_head=self.nheads,
            split=True, dropout=dropout, d_qk=embed_dim, compute_v=compute_v, use_argmax=self.use_argmax
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
            tensor=self._compute_unfolding_weights(self.kernel_size, self.stem_out_channels),
            persistent=False,
        )
        self.register_buffer(
            name="unfolding_weights_image",
            tensor=self._compute_unfolding_weights(self.kernel_size, 3),
            persistent=False,
        )
        self.register_buffer(
            name="unfolding_weights_mask",
            tensor=self._compute_unfolding_weights(self.kernel_size, 1),
            persistent=False,
        )

    def _compute_unfolding_weights(self, kernel_size, channels) -> torch.Tensor:
        weights = torch.eye(kernel_size *
                            kernel_size, dtype=torch.float)
        weights = weights.reshape(
            (kernel_size * kernel_size,
             1, kernel_size, kernel_size)
        )
        weights = weights.repeat(channels, 1, 1, 1)
        return weights

    def unfolding_coreml(self, feature_map: torch.Tensor, weights, kernel_size: int):
        # im2col is not implemented in Coreml, so here we hack its implementation using conv2d
        # we compute the weights
        batch_size, in_channels, img_h, img_w = feature_map.shape
        patches = F.conv2d(
            feature_map,
            weights,
            bias=None,
            stride=(kernel_size, kernel_size),
            padding=0,
            dilation=1,
            groups=in_channels,
        )
        return patches, (img_h, img_w)

    def folding_coreml(self, patches: torch.Tensor, output_size, kernel_size: int, use_final_conv: bool) -> torch.Tensor:
        # col2im is not supported on coreml, so tracing fails
        # We hack folding function via pixel_shuffle to enable coreml tracing
        if use_final_conv and self.final_conv:
            patches = rearrange(patches, 'b (p1 p2) c -> b c p1 p2', p1=self.window_size, p2=self.window_size)
            patches = self.final_conv(patches)
            patches = rearrange(patches, 'b c p1 p2 -> b (p1 p2) c')
        final_image = rearrange(patches, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                        h=output_size[0]//kernel_size, w=output_size[1]//kernel_size,
                        p1=kernel_size, p2=kernel_size)
        return final_image

    def forward(self, image, mask):
        image_coarse_inpainting, features = self.encoder_decoder(image)
        if self.mask_inpainting:
            image = image_coarse_inpainting*mask + image * (1 - mask)
        else:
            image = image_coarse_inpainting
        image_to_return = image_coarse_inpainting
        
        image_blurred = self.final_gaussian_blur(image)
        image_as_patches_blurred, _ = self.unfolding_coreml(
            image_blurred, self.unfolding_weights, self.kernel_size)

        image_as_patches, sizes = self.unfolding_coreml(
            image, self.unfolding_weights, self.kernel_size)
        
        image_as_patches = image_as_patches - image_as_patches_blurred

        pos = self.positionalencoding.repeat(
            image_as_patches.size(0), 1, 1).unsqueeze(2) if self.use_qpos else None

        mask_same_res_as_features_pooled, _ = self.unfolding_coreml(
                mask, self.unfolding_weights_mask, self.kernel_size)
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

        out = out - image_as_patches_blurred.flatten(start_dim=2).transpose(1, 2)

        mask = mask_same_res_as_features_pooled.squeeze(1).squeeze(-1).unsqueeze(-1)
        out = out * mask + image_as_patches * (1 - mask)

        out = self.folding_coreml(out, sizes, self.kernel_size, use_final_conv=True)


        return out, atten_weights, image_to_return

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
    def __init__(self, variant='s4', **kwargs):
        super().__init__()
        self.model = mobileone(variant=variant, **kwargs)

        # Decoder
        self.d4 = nn.ConvTranspose2d(2048, 1792, kernel_size=4, stride=2, padding=1)
        self.d3 = nn.ConvTranspose2d(1792 + 1792, 896, kernel_size=4, stride=2, padding=1)
        self.d2 = nn.ConvTranspose2d(896 + 896, 384, kernel_size=4, stride=2, padding=1)
        self.d1 = nn.ConvTranspose2d(384 + 384, 64, kernel_size=4, stride=2, padding=1)
        self.d0 = nn.ConvTranspose2d(64 + 64, 3, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = []
        x0 = self.model.stage0(x)
        features.append(x0)
        x1 = self.model.stage1(x0)
        features.append(x1)
        x2 = self.model.stage2(x1)
        features.append(x2)
        x3 = self.model.stage3(x2)
        features.append(x3)
        x4 = self.model.stage4(x3)
        features.append(x4)

        out = self.relu(self.d4(x4))
        out = torch.cat([out, x3], dim=1)
        out = self.relu(self.d3(out))
        out = torch.cat([out, x2], dim=1)
        out = self.relu(self.d2(out))
        out = torch.cat([out, x1], dim=1)
        out = self.relu(self.d1(out))
        out = torch.cat([out, x0], dim=1)
        out = self.sigmoid(self.d0(out))

        return out, features



class AttentionUpscaling(nn.Module):
    def __init__(self, patch_inpainting_module: "PatchInpainting"):
        super().__init__()
        self.patch_inpainting = patch_inpainting_module

    def forward(self, x_hr, x_lr_inpainted, attn_map):
        hr_h, hr_w = x_hr.shape[-2:]
        lr_h, lr_w = x_lr_inpainted.shape[-2:]
        
        # Upsample the low-resolution inpainted image
        x_hr_base = F.interpolate(x_lr_inpainted, size=(hr_h, hr_w), mode='bicubic', align_corners=False)

        # High-frequency token mixing
        hr_patch_size = self.patch_inpainting.kernel_size * (hr_h // lr_h)
        
        # Create HR patches
        unfolding_weights_hr = self.patch_inpainting._compute_unfolding_weights(kernel_size=hr_patch_size, channels=x_hr.shape[1])
        
        hr_patches, _ = self.patch_inpainting.unfolding_coreml(x_hr, unfolding_weights_hr, hr_patch_size)
        
        # Create blurred HR patches
        hr_blurred = self.patch_inpainting.final_gaussian_blur(x_hr)
        hr_patches_blurred, _ = self.patch_inpainting.unfolding_coreml(hr_blurred, unfolding_weights_hr, hr_patch_size)
        
        # Compute HR high-frequency patches
        hr_hf_patches = hr_patches - hr_patches_blurred
        hr_hf_patches = hr_hf_patches.flatten(start_dim=2).transpose(1, 2)
        
        # Weighted sum of HR high-frequency patches
        reconstructed_hr_hf_patches = torch.matmul(attn_map.squeeze(1), hr_hf_patches)
        
        # Fold the reconstructed patches
        reconstructed_hr_hf_image = self.patch_inpainting.folding_coreml(reconstructed_hr_hf_patches, (hr_h, hr_w), kernel_size=hr_patch_size, use_final_conv=False)
        
        # Add high-frequency details to the upsampled LR image
        final_hr_image = x_hr_base + reconstructed_hr_hf_image
        
        return final_hr_image


class InpaintingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.coarse_model = MobileOneCoarse(**config['coarse_model']['parameters'])
        self.generator = PatchInpainting(**config['generator']['params'], model=self.coarse_model)

    def forward(self, image, mask):
        return self.generator(image, mask)


if __name__ == '__main__':
    # The model is designed to work on a fixed low resolution (e.g., 512x512)
    # as described in the paper. The high-resolution inpainting is achieved
    # through a separate "Attention Upscaling" step which is not part of this model.
    config = {
        'coarse_model': {
            'class': 'MobileOneCoarse',
            'parameters': {
                'variant': 's4'
            }
        },
        'generator': {
            'generator_class': 'PatchInpainting',
            'params': {
                'kernel_size': 8,
                'nheads': 1,
                'stem_out_stride': 1,
                'stem_out_channels': 3,
                'merge_mode': 'all',
                'image_size': 512,
                'embed_dim': 576,
                'use_qpos': None,
                'use_kpos': None,
                'dropout': 0.1,
                'feature_i': 2,
                'concat_features': True,
                'final_conv': True,
                'feature_dim': 896,
                'attention_type': 'MultiHeadAttention',
                'compute_v': False,
                'use_argmax': False
            }
        }
    }

    # Define the high-resolution image size
    HR_IMAGE_SIZE = 2048

    # Create a high-resolution dummy image and mask
    high_res_image = torch.randn(1, 3, HR_IMAGE_SIZE, HR_IMAGE_SIZE)
    high_res_mask = torch.zeros(1, 1, HR_IMAGE_SIZE, HR_IMAGE_SIZE)
    high_res_mask[:, :, HR_IMAGE_SIZE//2:, HR_IMAGE_SIZE//2:] = 1

    # Downsample inputs for the generator, as described in the paper
    low_res_image = F.interpolate(high_res_image, size=512, mode='bicubic', antialias=True)
    low_res_mask = F.interpolate(high_res_mask, size=512)
    masked_low_res_image = low_res_image * (1 - low_res_mask)

    # Instantiate the model
    model = InpaintingModel(config)

    # Forward pass with the downsampled image
    output, attn_scores, temp_image = model(masked_low_res_image, low_res_mask)

    # The output of the model is at the low resolution (512x512)
    print(f"Low resolution output shape: {output.shape}")

    # Apply attention upscaling
    attention_upscaler = AttentionUpscaling(model.generator)
    final_output = attention_upscaler(high_res_image, output, attn_scores)
    
    print(f"Final high resolution output shape: {final_output.shape}")
    assert final_output.shape == high_res_image.shape
    print("Forward pass with attention upscaling completed successfully.")
