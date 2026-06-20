"""
denoiser/model.py

Noise2Noise 1D U-Net with Squeeze-and-Excitation attention blocks.

Architecture overview:
  Input  [B, 1, 1000]
    → Encoder ×4  (conv pair + SE + MaxPool — channels double each level)
    → Bottleneck   (conv pair + Dropout)
    → Decoder ×4  (ConvTranspose + skip concat + conv pair + SE)
    → Output head  (1×1 conv → single flux channel)
  Output [B, 1, 1000]

Key design decisions:
  - Skip connections saved BEFORE MaxPool — preserves fine-grained temporal detail
  - SE blocks re-weight channels by learned importance — model focuses on
    transit-relevant frequencies, suppresses noise-dominant channels
  - Dropout only in bottleneck — decoder spatial reconstruction stays stable
  - bias=False on all Conv+BN pairs — BN subsumes the bias term
  - F.pad in DecoderBlock handles the 125→124 odd-dimension mismatch at dec4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG


# ─────────────────────────────────────────────────────────────────────────────
# SE Block — Squeeze-and-Excitation channel attention
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al. 2018) adapted for 1D signals.

    Learns a per-channel weight vector — channels carrying transit-relevant
    frequency content get amplified; noise-dominant channels get suppressed.

    Steps:
      1. Squeeze  : GlobalAvgPool over time axis → [B, C]
      2. Excite   : Two-layer MLP with bottleneck → [B, C] weights in (0, 1)
      3. Scale    : Multiply input channels by learned weights

    Args:
        channels  : number of input/output channels C
        reduction : bottleneck reduction ratio r (default 16)
                    bottleneck hidden dim = C // r

    Shape:
        Input  : [B, C, T]
        Output : [B, C, T]  (same shape, channels re-weighted)
    """

    def __init__(self, channels: int, reduction: int = CFG.se_reduction):
        super().__init__()

        assert channels % reduction == 0, (
            f"SEBlock: channels ({channels}) must be divisible by "
            f"reduction ({reduction})"
        )

        hidden = channels // reduction

        self.squeeze  = nn.AdaptiveAvgPool1d(1)   # [B, C, T] → [B, C, 1]

        self.excitation = nn.Sequential(
            nn.Flatten(),                           # [B, C, 1] → [B, C]
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),                           # weights in (0, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        w = self.squeeze(x)            # [B, C, 1]
        w = self.excitation(w)         # [B, C]
        w = w.unsqueeze(-1)            # [B, C, 1] — broadcast over time
        return x * w                   # [B, C, T] — channel-wise scale


# ─────────────────────────────────────────────────────────────────────────────
# Encoder Block
# ─────────────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """
    One level of the U-Net encoder.

    Applies two Conv-BN-ReLU pairs followed by an SE block, then MaxPool.
    The skip connection is saved BEFORE MaxPool — it retains full temporal
    resolution for the corresponding decoder level to use.

    Args:
        in_ch        : input channels
        out_ch       : output channels (doubles at each encoder level)
        se_reduction : SE bottleneck reduction ratio

    Forward returns:
        pooled : [B, out_ch, T//2]  — passed to next encoder level
        skip   : [B, out_ch, T]     — saved for decoder concat
    """

    def __init__(
        self,
        in_ch:        int,
        out_ch:       int,
        se_reduction: int = CFG.se_reduction,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            # First conv
            nn.Conv1d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            # Second conv
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.se   = SEBlock(out_ch, se_reduction)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x    = self.conv_block(x)   # [B, out_ch, T]
        x    = self.se(x)           # [B, out_ch, T]
        skip = x                    # save BEFORE pool
        x    = self.pool(x)         # [B, out_ch, T//2]
        return x, skip


# ─────────────────────────────────────────────────────────────────────────────
# Decoder Block
# ─────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    One level of the U-Net decoder.

    Upsamples with ConvTranspose1d, concatenates the encoder skip connection,
    applies two Conv-BN-ReLU pairs, then an SE block.

    Size mismatch handling:
      When the encoder had an odd-length temporal dimension (e.g. 125),
      MaxPool floors it to 62. ConvTranspose then produces 124, not 125.
      We pad the upsampled tensor by 1 on the right to match the skip size.
      This is safe — the padded value is 0, and the subsequent convolution
      integrates the context from surrounding valid positions.

    Args:
        in_ch        : channels coming from previous decoder level (or bottleneck)
        out_ch       : output channels (halves at each decoder level)
        se_reduction : SE bottleneck reduction ratio

    Forward args:
        x    : [B, in_ch,  T_low]  — from previous decoder / bottleneck
        skip : [B, out_ch, T_high] — from corresponding encoder level

    Forward returns:
        [B, out_ch, T_high]
    """

    def __init__(
        self,
        in_ch:        int,
        out_ch:       int,
        se_reduction: int = CFG.se_reduction,
    ):
        super().__init__()

        # Upsample: in_ch → out_ch, doubles temporal resolution
        self.upsample = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=2, stride=2
        )

        # After concat with skip: out_ch (upsample) + out_ch (skip) = out_ch * 2
        self.conv_block = nn.Sequential(
            nn.Conv1d(out_ch * 2, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch,     out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.se = SEBlock(out_ch, se_reduction)

    def forward(
        self,
        x:    torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x)                   # [B, out_ch, T_low * 2]

        # Handle odd-dimension mismatch (e.g. dec4: 124 vs skip 125)
        if x.size(2) != skip.size(2):
            diff = skip.size(2) - x.size(2)    # always +1 in our architecture
            x = F.pad(x, (0, diff))            # pad right side only

        x = torch.cat([x, skip], dim=1)        # [B, out_ch*2, T_high]
        x = self.conv_block(x)                 # [B, out_ch, T_high]
        x = self.se(x)                         # [B, out_ch, T_high]
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck
# ─────────────────────────────────────────────────────────────────────────────

class Bottleneck(nn.Module):
    """
    U-Net bottleneck — deepest representation, no skip connection here.

    Two Conv-BN-ReLU pairs followed by Dropout.
    Dropout only here (not in encoder/decoder) — keeps reconstruction
    quality stable while still regularising the latent representation.

    Shape: [B, C*8, T/16] → [B, C*16, T/16]
    """

    def __init__(
        self,
        in_ch:   int,
        out_ch:  int,
        dropout: float = CFG.bottleneck_dropout,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full Noise2Noise U-Net
# ─────────────────────────────────────────────────────────────────────────────

class NoiseToNoiseUNet(nn.Module):
    """
    Full 1D U-Net Noise2Noise denoiser for TESS light curves.

    Channel progression (base_channels C = 64):
      Encoder:     1 → C → C*2 → C*4 → C*8     (64 → 128 → 256 → 512)
      Bottleneck:  C*8 → C*16                   (512 → 1024)
      Decoder:     C*16 → C*8 → C*4 → C*2 → C  (1024 → 512 → 256 → 128 → 64)
      Output head: C → 1                         (64 → 1)

    Temporal resolution:
      Input:      T = 1000
      After enc1: T/2  = 500
      After enc2: T/4  = 250
      After enc3: T/8  = 125
      After enc4: T/16 = 62   ← 125 is odd, MaxPool floors to 62
      Bottleneck: 62
      After dec4: 124  → padded to 125 before concat with skip_4
      After dec3: 250
      After dec2: 500
      After dec1: 1000
      Output:     1000

    Args:
        base_channels       : channels in first encoder block (default 64)
        se_reduction        : SE attention reduction ratio (default 16)
        bottleneck_dropout  : dropout probability in bottleneck (default 0.1)

    Shape:
        Input  : [B, 1, T]  — T must be divisible by 16 for clean dims,
                              or handled gracefully by F.pad in decoders.
                              Recommended T = 1000 (CFG.input_length)
        Output : [B, 1, T]  — denoised flux, same shape as input

    MC Dropout usage:
        For uncertainty estimation at inference time, call model.train()
        to keep dropout active, then run N forward passes.
        See pipeline/uncertainty.py for the wrapper.

    Example:
        model = NoiseToNoiseUNet()
        x = torch.randn(4, 1, 1000)
        out = model(x)   # [4, 1, 1000]
    """

    def __init__(
        self,
        base_channels:      int   = CFG.base_channels,
        se_reduction:       int   = CFG.se_reduction,
        bottleneck_dropout: float = CFG.bottleneck_dropout,
    ):
        super().__init__()

        C = base_channels   # 64 by default

        # ── Encoder ───────────────────────────────────────────────────────
        # Each block: [B, in, T] → pooled:[B, out, T//2] + skip:[B, out, T]
        self.enc1 = EncoderBlock(1,    C,    se_reduction)   # 1   → 64
        self.enc2 = EncoderBlock(C,    C*2,  se_reduction)   # 64  → 128
        self.enc3 = EncoderBlock(C*2,  C*4,  se_reduction)   # 128 → 256
        self.enc4 = EncoderBlock(C*4,  C*8,  se_reduction)   # 256 → 512

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = Bottleneck(C*8, C*16, bottleneck_dropout)  # 512 → 1024

        # ── Decoder ───────────────────────────────────────────────────────
        # Each block: [B, in, T_low] + skip → [B, out, T_high]
        self.dec4 = DecoderBlock(C*16, C*8,  se_reduction)   # 1024 → 512
        self.dec3 = DecoderBlock(C*8,  C*4,  se_reduction)   # 512  → 256
        self.dec2 = DecoderBlock(C*4,  C*2,  se_reduction)   # 256  → 128
        self.dec1 = DecoderBlock(C*2,  C,    se_reduction)   # 128  → 64

        # ── Output head ───────────────────────────────────────────────────
        # 1×1 conv maps 64 channels → 1 flux channel
        # No activation — output is unbounded flux residual
        self.output_head = nn.Conv1d(C, 1, kernel_size=1, bias=True)

        # ── Weight initialisation ─────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming normal for Conv layers (optimal for ReLU).
        Standard init for BN (weight=1, bias=0).
        Xavier uniform for SE linear layers.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 1, T] — noisy input light curve

        Returns:
            [B, 1, T] — denoised light curve
        """
        # ── Encode ────────────────────────────────────────────────────────
        x, skip1 = self.enc1(x)    # x:[B,64,500]   skip1:[B,64,1000]
        x, skip2 = self.enc2(x)    # x:[B,128,250]  skip2:[B,128,500]
        x, skip3 = self.enc3(x)    # x:[B,256,125]  skip3:[B,256,250]
        x, skip4 = self.enc4(x)    # x:[B,512,62]   skip4:[B,512,125]

        # ── Bottleneck ────────────────────────────────────────────────────
        x = self.bottleneck(x)     # [B,1024,62]

        # ── Decode ────────────────────────────────────────────────────────
        x = self.dec4(x, skip4)    # [B,512,125]  (124→125 via F.pad)
        x = self.dec3(x, skip3)    # [B,256,250]
        x = self.dec2(x, skip2)    # [B,128,500]
        x = self.dec1(x, skip1)    # [B,64,1000]

        # ── Output ────────────────────────────────────────────────────────
        x = self.output_head(x)    # [B,1,1000]
        return x

    # ── Utility methods ───────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def verify_shapes(self, T: int = CFG.input_length) -> None:
        """
        Run a dummy forward pass and print shapes at every stage.
        Call this after building the model to confirm architecture is correct.

        Args:
            T : temporal length to test with (default 1000)
        """
        self.eval()
        with torch.no_grad():
            x = torch.randn(2, 1, T)
            print(f"\n{'='*55}")
            print(f"  NoiseToNoiseUNet shape verification  (T={T})")
            print(f"{'='*55}")
            print(f"  Input          : {list(x.shape)}")

            x_enc, skip1 = self.enc1(x)
            print(f"  After enc1     : {list(x_enc.shape)}   skip1: {list(skip1.shape)}")

            x_enc, skip2 = self.enc2(x_enc)
            print(f"  After enc2     : {list(x_enc.shape)}  skip2: {list(skip2.shape)}")

            x_enc, skip3 = self.enc3(x_enc)
            print(f"  After enc3     : {list(x_enc.shape)}  skip3: {list(skip3.shape)}")

            x_enc, skip4 = self.enc4(x_enc)
            print(f"  After enc4     : {list(x_enc.shape)}   skip4: {list(skip4.shape)}")

            x_bn = self.bottleneck(x_enc)
            print(f"  Bottleneck     : {list(x_bn.shape)}")

            x_dec = self.dec4(x_bn, skip4)
            print(f"  After dec4     : {list(x_dec.shape)}")

            x_dec = self.dec3(x_dec, skip3)
            print(f"  After dec3     : {list(x_dec.shape)}")

            x_dec = self.dec2(x_dec, skip2)
            print(f"  After dec2     : {list(x_dec.shape)}")

            x_dec = self.dec1(x_dec, skip1)
            print(f"  After dec1     : {list(x_dec.shape)}")

            out = self.output_head(x_dec)
            print(f"  Output         : {list(out.shape)}")
            print(f"{'='*55}")
            print(f"  Parameters     : {self.count_parameters():,}")
            print(f"{'='*55}\n")

            # Critical assertion
            assert out.shape == (2, 1, T), (
                f"Output shape {list(out.shape)} != expected [2, 1, {T}]"
            )
            print("  ✅ Shape verification passed\n")
