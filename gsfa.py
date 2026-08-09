"""
GSFA: Gated Spatial-Frequency Adapter

Fine-tunes SAM's image encoder by applying low-rank adaptation
to both spatial and frequency domain features, fused via a
gating mechanism.

From the paper: "GSFA introduces frequency-domain awareness
into the fine-tuning process. It jointly captures spatial and
frequency features and integrates them via a gating mechanism."
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter


class FrequencyDomainProcessor(nn.Module):
    """
    Low-rank adaptation applied in the frequency domain.

    Transforms token features via 2D FFT, then applies independent
    linear projections to real and imaginary components (rank=4),
    and transforms back via IFFT.
    """
    def __init__(self, input_channels):
        super(FrequencyDomainProcessor, self).__init__()

        # Low-rank reduction for real and imaginary parts
        self.real_reduce = nn.Linear(input_channels, 4)
        self.imag_reduce = nn.Linear(input_channels, 4)

        # Low-rank expansion for real and imaginary parts
        self.real_expand = nn.Linear(4, input_channels)
        self.imag_expand = nn.Linear(4, input_channels)

    def forward(self, x):
        """
        Args:
            x: [batch, height, width, channels] token features
        Returns:
            [batch, height, width, channels] frequency-adapted features
        """
        original_shape = x.shape

        # Transform to frequency domain
        x = x.permute(0, 3, 1, 2)  # [b, c, h, w]
        freq = torch.fft.rfft2(x)

        # Separate real and imaginary
        real = freq.real
        imag = freq.imag

        # Move channels to last dim: [b, c, h, w] -> [b, h, w, c]
        real = real.permute(0, 2, 3, 1)
        imag = imag.permute(0, 2, 3, 1)

        # Low-rank adaptation (Eq. 2, 3 in paper)
        real_reduced = self.real_reduce(real)
        imag_reduced = self.imag_reduce(imag)

        real_expanded = self.real_expand(real_reduced)
        imag_expanded = self.imag_expand(imag_reduced)

        # Transform back to [b, c, h, w]
        real_expanded = real_expanded.permute(0, 3, 1, 2)
        imag_expanded = imag_expanded.permute(0, 3, 1, 2)

        # Reconstruct complex tensor and apply IFFT (Eq. 4 in paper)
        freq_processed = torch.complex(real_expanded, imag_expanded)
        x_processed = torch.fft.irfft2(freq_processed, s=(original_shape[1], original_shape[2]))

        # Transform back to [b, h, w, c]
        x_processed = x_processed.permute(0, 2, 3, 1)

        return x_processed


class GatedFusion(nn.Module):
    """
    GF: Gated Fusion of spatial and frequency domain features (Eq. 5-7).

    Concatenates spatial and frequency features, passes through
    a 1x1 convolution + sigmoid to generate a spatial attention map (alpha),
    then fuses: alpha * spatial + (1-alpha) * frequency.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        """
        Args:
            x1: spatial-domain features [b, h, w, c]
            x2: frequency-domain features [b, h, w, c]
        Returns:
            Fused features [b, h, w, c]
        """
        # Convert to channel-first format
        x1_cf = x1.permute(0, 3, 1, 2)  # [b, c, h, w]
        x2_cf = x2.permute(0, 3, 1, 2)  # [b, c, h, w]

        # Generate spatial attention map alpha (Eq. 5)
        x_cat = torch.cat([x1_cf, x2_cf], dim=1)  # [b, 2c, h, w]
        weights = self.conv(x_cat)  # [b, 1, h, w]

        # Convert back to channel-last
        weights = weights.permute(0, 2, 3, 1)  # [b, h, w, 1]

        # Gated fusion (Eq. 6)
        return weights * x1 + (1 - weights) * x2


class _GSFA_qkv(nn.Module):
    """
    GSFA-augmented QKV layer injected into SAM's image encoder attention blocks.

    Replaces the standard qkv linear layer with:
    - Spatial-domain LoRA for Q and V (standard low-rank, Eq. 1)
    - Frequency-domain LoRA for Q and V (via FrequencyDomainProcessor, Eq. 2-4)
    - GatedFusion of spatial and frequency adaptations (Eq. 5-7)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
            linear_fq: nn.Module,
            linear_fv: nn.Module,
            gate_q: nn.Module,
            gate_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.linear_fq = linear_fq
        self.linear_fv = linear_fv
        self.gate_q = gate_q
        self.gate_v = gate_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)  # B, N, 3*org_C

        # Frequency-domain adaptation
        fq = self.linear_fq(x)
        fv = self.linear_fv(x)

        # Spatial-domain adaptation (standard LoRA, Eq. 1 in paper)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))

        # Gated fusion of spatial and frequency adaptations
        qq = self.gate_q(new_q, fq)
        vv = self.gate_v(new_v, fv)

        # Add to original QKV output
        qkv[:, :, :, : self.dim] += qq
        qkv[:, :, :, -self.dim:] += vv

        return qkv


class GSFA(nn.Module):
    """
    GSFA: Gated Spatial-Frequency Adapter for SAM's image encoder.

    Applies low-rank adaptation separately to spatial and frequency
    components of token features, then integrates them via gated fusion.
    Injected into each transformer block's QKV attention layer.

    Args:
        sam_model: a SAM model instance (e.g. CoroSAM)
        r: rank of LoRA (default 4, following the paper)
        lora_layer: which transformer layers to apply GSFA to
    """

    def __init__(self, sam_model: nn.Module, r: int, lora_layer=None):
        super(GSFA, self).__init__()

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))

        self.w_As = []
        self.w_Bs = []

        # Freeze image encoder first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Apply GSFA to selected attention blocks
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features

            # Spatial LoRA components
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)

            # Frequency domain processors
            fq = FrequencyDomainProcessor(self.dim)
            fv = FrequencyDomainProcessor(self.dim)

            # Gating modules
            gq = GatedFusion(self.dim)
            gv = GatedFusion(self.dim)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            # Replace QKV with GSFA-augmented version
            blk.attn.qkv = _GSFA_qkv(
                w_qkv_linear,
                w_a_linear_q, w_b_linear_q,
                w_a_linear_v, w_b_linear_v,
                fq, fv, gq, gv,
            )

        self.reset_parameters()
        self.sam = sam_model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output, image_size):
        return self.sam(batched_input, multimask_output, image_size)
