"""
classifier/model.py

CNN-LSTM six-class transit signal classifier.

Architecture overview:
  Input  [B, 1, 200]                      ← phase-folded light curve segment
    → CNN Feature Extractor × 3            (local pattern detection: shape, depth, symmetry)
    → Permute [B, 128, 25] → [B, 25, 128]
    → BiLSTM × 2                           (temporal order: secondary eclipse, odd/even depth)
    → Attention Pooling                     (focus on transit centre vs flat regions)
    → Classification Head                   (three linear layers → six-class output)
  Output [B, 6]                            ← raw logits (apply softmax for probabilities)

Key design decisions:
  - CNN before LSTM: CNN extracts local features (ingress/egress shape, transit depth,
    symmetry) from the folded profile. LSTM then models temporal structure across the
    25-step condensed sequence — critical for catching secondary eclipses (EB signature)
    and odd/even depth alternation (HEB signature).

  - Bidirectional LSTM: processes the folded curve left-to-right AND right-to-left.
    Secondary eclipse at phase 0.5 is equally identifiable from either direction.

  - Decreasing kernel sizes (7→5→3): first block captures broad transit envelope,
    subsequent blocks refine ingress/egress shape details.

  - Dropout after FIRST conv only in each CNN block (spec-exact). Second conv
    reconstructs spatial coherence — dropout there would degrade learned features.

  - Attention pooling: learns which of the 25 sequence positions matter most.
    Transit centre gets high weight; flat out-of-transit baseline gets low weight.
    Weights stored in attn.last_weights after every forward pass — used by dashboard.

  - forward() returns raw logits — CrossEntropyLoss applies log-softmax internally.
    Never pass logits through F.softmax before the loss. Use predict_proba() for
    softmax output at inference time.

  - MC Dropout: call enable_mc_dropout() before inference. Sets BN to eval mode
    (uses learned running statistics) while keeping all Dropout layers active.
    Standard model.eval() would disable Dropout — do NOT use that for MC inference.

  - Weight init: Kaiming normal for Conv (optimal for ReLU), Xavier uniform for
    Linear, Orthogonal for LSTM weight_hh (prevents vanishing gradients in RNNs),
    forget gate bias initialised to 1.0 (standard LSTM trick for long sequences).

Shape trace (T=200):
  Input        [B, 1,   200]
  cnn1         [B, 32,  100]
  cnn2         [B, 64,   50]
  cnn3         [B, 128,  25]
  permute      [B, 25,  128]
  lstm1        [B, 25,  256]   ← 128 forward + 128 backward
  lstm2        [B, 25,  128]   ← 64 forward  + 64 backward
  attention    [B,      128]   ← weighted sum over seq dim
  head         [B,        6]   ← raw logits
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CNN1DBlock
# ─────────────────────────────────────────────────────────────────────────────

class CNN1DBlock(nn.Module):
    """
    One CNN feature extraction block.

    Pattern:
      Conv → BN → ReLU → Dropout    ← first conv (with dropout)
      Conv → BN → ReLU              ← second conv (no dropout — preserves coherence)
      MaxPool(2)                     ← halves temporal dimension

    Dropout placement rationale:
      Dropout after the first conv encourages the block to learn redundant representations
      of broad features (transit envelope, baseline level). The second conv then integrates
      these noisy intermediate features into a clean local descriptor. Applying dropout after
      the second conv would corrupt the spatial reconstruction the block just completed.

    Shape:
      Input  : [B, in_ch,  T]
      Output : [B, out_ch, T//2]

    Args:
        in_ch       : input channels
        out_ch      : output channels
        kernel_size : conv kernel size (use odd numbers for symmetric padding)
        dropout     : Dropout probability after first conv (default 0.1)
    """

    def __init__(
        self,
        in_ch:       int,
        out_ch:      int,
        kernel_size: int,
        dropout:     float = CFG.cnn_dropout,
    ):
        super().__init__()

        # "same" padding: output length = input length before pooling
        padding = kernel_size // 2

        # First conv: with Dropout after ReLU
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Second conv: no Dropout
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

        # MaxPool halves temporal dimension
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)   # [B, out_ch, T]
        x = self.conv2(x)   # [B, out_ch, T]
        x = self.pool(x)    # [B, out_ch, T//2]
        return x


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTMBlock
# ─────────────────────────────────────────────────────────────────────────────

class BiLSTMBlock(nn.Module):
    """
    Single bidirectional LSTM layer followed by Dropout.

    Wraps nn.LSTM (num_layers=1, bidirectional=True, batch_first=True).
    Dropout is applied on the output — this is deliberate. Using the built-in
    LSTM dropout parameter would NOT apply dropout on the final layer's output,
    which is what we want here.

    Shape:
      Input  : [B, seq_len, input_size]
      Output : [B, seq_len, hidden_size * 2]   ← 2 = bidirectional

    Args:
        input_size  : feature dimension of input sequence
        hidden_size : LSTM hidden units (output dim = hidden_size * 2)
        dropout     : Dropout probability applied on LSTM output (default 0.3)
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        dropout:     float = CFG.lstm_dropout,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,       # input/output: [B, seq, features]
            bias=True,
        )
        self.dropout    = nn.Dropout(p=dropout)
        self.output_dim = hidden_size * 2   # expose for downstream size calculation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x       : [B, seq_len, input_size]
        # out     : [B, seq_len, hidden_size*2]
        # (h_n, c_n) discarded — we only need the full sequence output
        out, _ = self.lstm(x)
        out    = self.dropout(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# AttentionPool
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """
    Soft attention pooling over the sequence dimension.

    Replaces naive mean/max pooling. Learns to focus on transit-relevant
    timesteps (transit centre, ingress/egress edges) rather than treating
    all 25 positions equally.

    Mechanism:
      score[t] = Linear(h[t])          ← unnormalised importance per timestep
      weight[t] = softmax(score)[t]    ← normalised attention weight, sums to 1
      context  = Σ_t weight[t] * h[t] ← weighted sum = attended context vector

    Attention weights are stored in self.last_weights after every forward pass
    (detached from grad graph). The dashboard reads these to show which parts of
    the folded light curve the model focused on — no separate forward pass needed.

    Shape:
      Input  : [B, seq_len, hidden_dim]
      Output : [B, hidden_dim]

    Args:
        hidden_dim : feature dimension of incoming sequence (BiLSTM output dim)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score          = nn.Linear(hidden_dim, 1, bias=True)
        self.last_weights: Optional[torch.Tensor] = None   # [B, seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x       : [B, seq_len, hidden_dim]
        scores  = self.score(x).squeeze(-1)          # [B, seq_len]
        weights = F.softmax(scores, dim=1)            # [B, seq_len] — sums to 1
        self.last_weights = weights.detach()          # store for inspection, no grad

        # Weighted sum over sequence: [B, seq_len, hidden_dim] × [B, seq_len, 1]
        context = (x * weights.unsqueeze(-1)).sum(dim=1)   # [B, hidden_dim]
        return context


# ─────────────────────────────────────────────────────────────────────────────
# Head builder (module-level helper)
# ─────────────────────────────────────────────────────────────────────────────

def _build_head(
    input_dim:    int,
    hidden_sizes: list,
    dropouts:     list,
    num_classes:  int,
) -> nn.Sequential:
    """
    Build classification head dynamically from config lists.

    Pattern for each hidden layer: Linear → ReLU → Dropout
    Final layer: Linear only (no activation, no dropout — raw logits)

    Args:
        input_dim    : size of attention context vector (128)
        hidden_sizes : list of hidden layer sizes  e.g. [64, 32]
        dropouts     : list of dropout rates       e.g. [0.5, 0.3]
        num_classes  : number of output classes    (6)

    Returns:
        nn.Sequential that maps [B, input_dim] → [B, num_classes]
    """
    assert len(hidden_sizes) == len(dropouts), (
        f"head_hidden_sizes ({len(hidden_sizes)}) and "
        f"head_dropouts ({len(dropouts)}) must be same length"
    )

    layers   = []
    prev_dim = input_dim

    for h_dim, drop in zip(hidden_sizes, dropouts):
        layers += [
            nn.Linear(prev_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop),
        ]
        prev_dim = h_dim

    # Final projection to logits — NO softmax here
    # CrossEntropyLoss applies log-softmax internally
    # Use predict_proba() or F.softmax(logits, dim=-1) for probabilities
    layers.append(nn.Linear(prev_dim, num_classes))

    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# TransitClassifier — full model
# ─────────────────────────────────────────────────────────────────────────────

class TransitClassifier(nn.Module):
    """
    CNN-LSTM six-class classifier for exoplanet transit signal discrimination.

    Input:  phase-folded light curve segment — [B, 1, 200]
    Output: raw class logits               — [B, 6]

    Six classes:
      0 — Planet Transit (PT)
      1 — Eclipsing Binary (EB)
      2 — Background Eclipsing Binary (BEB)
      3 — Hierarchical Eclipsing Binary (HEB)
      4 — Stellar Variability (SV)
      5 — Instrumental Artifact (IA)

    Standard usage:
      model = TransitClassifier()
      logits = model(x)                           # training — pass to CrossEntropyLoss
      probs  = model.predict_proba(x)             # inference — softmax applied

    MC Dropout inference:
      model.enable_mc_dropout()                   # BN frozen, Dropout active
      with torch.no_grad():
          preds = [F.softmax(model(x), dim=-1) for _ in range(50)]
      mean = torch.stack(preds).mean(0)           # [B, 6]
      std  = torch.stack(preds).std(0)            # [B, 6]

    Phase 2 fine-tuning (freeze CNN, retrain LSTM+head):
      model.load_state_dict(torch.load("classifier_kepler.pth")["model_state"])
      model.freeze_cnn()
      # optimiser only sees LSTM + attention + head parameters

    Hyderabad fine-tuning (all layers):
      model.unfreeze_all()
      # optimiser sees all parameters, lower lr
    """

    def __init__(
        self,
        num_classes:       int   = CFG.num_classes,
        input_channels:    int   = CFG.input_channels,
        cnn_channels:      list  = None,
        cnn_kernels:       list  = None,
        cnn_dropout:       float = CFG.cnn_dropout,
        lstm_hidden_sizes: list  = None,
        lstm_dropout:      float = CFG.lstm_dropout,
        head_hidden_sizes: list  = None,
        head_dropouts:     list  = None,
    ):
        super().__init__()

        # ── Resolve defaults from CFG ──────────────────────────────────────
        if cnn_channels      is None: cnn_channels      = list(CFG.cnn_channels)
        if cnn_kernels       is None: cnn_kernels        = list(CFG.cnn_kernels)
        if lstm_hidden_sizes is None: lstm_hidden_sizes  = list(CFG.lstm_hidden_sizes)
        if head_hidden_sizes is None: head_hidden_sizes  = list(CFG.head_hidden_sizes)
        if head_dropouts     is None: head_dropouts      = list(CFG.head_dropouts)

        assert len(cnn_channels) == len(cnn_kernels) == 3, (
            "Exactly 3 CNN blocks required — "
            f"got {len(cnn_channels)} channels and {len(cnn_kernels)} kernels"
        )
        assert len(lstm_hidden_sizes) == 2, (
            f"Exactly 2 BiLSTM layers required — got {len(lstm_hidden_sizes)}"
        )

        # ── Store for freeze methods and verify_shapes ─────────────────────
        self._cnn_channels      = cnn_channels
        self._lstm_hidden_sizes = lstm_hidden_sizes
        self._num_classes       = num_classes

        # ── CNN Feature Extractor ──────────────────────────────────────────
        #   Block 0: input_channels   → cnn_channels[0]   (e.g. 1  → 32)
        #   Block 1: cnn_channels[0]  → cnn_channels[1]   (e.g. 32 → 64)
        #   Block 2: cnn_channels[1]  → cnn_channels[2]   (e.g. 64 → 128)
        in_channels = [input_channels] + cnn_channels[:-1]

        self.cnn1 = CNN1DBlock(in_channels[0], cnn_channels[0], cnn_kernels[0], cnn_dropout)
        self.cnn2 = CNN1DBlock(in_channels[1], cnn_channels[1], cnn_kernels[1], cnn_dropout)
        self.cnn3 = CNN1DBlock(in_channels[2], cnn_channels[2], cnn_kernels[2], cnn_dropout)

        # ── BiLSTM Sequential Extractor ────────────────────────────────────
        #   lstm1 input  : cnn_channels[-1]          = 128
        #   lstm1 output : lstm_hidden_sizes[0] * 2  = 256  (bidirectional)
        #   lstm2 input  : lstm_hidden_sizes[0] * 2  = 256
        #   lstm2 output : lstm_hidden_sizes[1] * 2  = 128  (bidirectional)
        self.lstm1 = BiLSTMBlock(
            input_size  = cnn_channels[-1],
            hidden_size = lstm_hidden_sizes[0],
            dropout     = lstm_dropout,
        )
        self.lstm2 = BiLSTMBlock(
            input_size  = self.lstm1.output_dim,      # 256
            hidden_size = lstm_hidden_sizes[1],
            dropout     = lstm_dropout,
        )

        # ── Attention Pooling ──────────────────────────────────────────────
        #   input/output dim : lstm2.output_dim = 128
        self.attn = AttentionPool(hidden_dim=self.lstm2.output_dim)

        # ── Classification Head ────────────────────────────────────────────
        #   input_dim : self.lstm2.output_dim = 128
        #   hidden    : [64, 32]   with dropouts [0.5, 0.3]
        #   output    : num_classes = 6  (raw logits)
        self.head = _build_head(
            input_dim    = self.lstm2.output_dim,
            hidden_sizes = head_hidden_sizes,
            dropouts     = head_dropouts,
            num_classes  = num_classes,
        )

        # ── Weight initialisation ──────────────────────────────────────────
        self._init_weights()

        logger.info(
            f"TransitClassifier built — "
            f"{self.count_parameters():,} parameters total"
        )

    # ── Weight initialisation ──────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Per-layer weight initialisation.

          Conv1d       : Kaiming normal (fan_out, ReLU) — optimal for deep conv stacks
          BatchNorm1d  : weight=1, bias=0 (standard)
          Linear       : Xavier uniform — balanced variance for classification layers
          LSTM
            weight_ih  : Xavier uniform — input projection, no activation after it
            weight_hh  : Orthogonal — prevents vanishing/exploding gradients in RNNs
            bias        : zeros, except forget gate bias set to 1.0
                          Forget gate init=1 keeps gradients flowing through time
                          early in training (Jozefowicz et al. 2015).
        """
        for m in self.modules():

            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        # Input-hidden: Xavier uniform
                        nn.init.xavier_uniform_(param.data)

                    elif "weight_hh" in name:
                        # Hidden-hidden: Orthogonal — critical for RNN stability
                        # Orthogonal matrices preserve gradient norm through time
                        nn.init.orthogonal_(param.data)

                    elif "bias" in name:
                        nn.init.zeros_(param.data)
                        # Forget gate lives at [hidden_size : 2*hidden_size]
                        # LSTM gate order: input | forget | cell | output
                        # Initialising forget bias to 1.0 → sigmoid(1) ≈ 0.73
                        # → model starts with a slight bias to remember context
                        n      = param.size(0)
                        hidden = n // 4
                        param.data[hidden : 2 * hidden].fill_(1.0)

    # ── Forward pass ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 1, T]  where T = 200 (phase-folded segment, normalised flux)

        Returns:
            logits : [B, 6]  — raw unnormalised class scores
                     Apply F.softmax(logits, dim=-1) for probabilities.
                     Do NOT apply softmax before passing to CrossEntropyLoss.

        Side effect:
            self.attn.last_weights is updated with [B, 25] attention weights
            after every forward pass. Access with model.attn.last_weights.
        """
        # ── CNN: local pattern detection ──────────────────────────────────
        # [B, 1, 200] → [B, 32, 100] → [B, 64, 50] → [B, 128, 25]
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)

        # Reshape for LSTM: channels → features, spatial → sequence
        # [B, 128, 25] → [B, 25, 128]
        # .contiguous() ensures LSTM gets a contiguous memory layout
        x = x.permute(0, 2, 1).contiguous()

        # ── BiLSTM: temporal sequence modelling ───────────────────────────
        # [B, 25, 128] → [B, 25, 256]
        x = self.lstm1(x)
        # [B, 25, 256] → [B, 25, 128]
        x = self.lstm2(x)

        # ── Attention pooling: collapse seq → vector ───────────────────────
        # [B, 25, 128] → [B, 128]
        x = self.attn(x)

        # ── Classification head: logits ────────────────────────────────────
        # [B, 128] → [B, 64] → [B, 32] → [B, 6]
        x = self.head(x)

        return x   # raw logits

    # ── Inference helpers ──────────────────────────────────────────────────

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard (non-MC) inference. Returns softmax class probabilities.

        Handles input shape variants automatically:
          [T]      → unsqueezed to [1, 1, T]
          [B, T]   → unsqueezed to [B, 1, T]
          [B, 1, T]→ used as-is

        Args:
            x : any of the above shapes

        Returns:
            probs : [B, 6]  float32, sums to 1 per sample
                    probs[b, c] = probability that sample b belongs to class c
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)   # [T] → [1, 1, T]
        elif x.dim() == 2:
            x = x.unsqueeze(1)                 # [B, T] → [B, 1, T]

        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def enable_mc_dropout(self) -> None:
        """
        Enable MC Dropout for uncertainty estimation at inference time.

        Problem:
          model.eval() disables ALL stochastic behaviour including Dropout.
          For MC Dropout we need BN frozen (eval stats) but Dropout active.

        Solution:
          1. Call model.eval() → freezes BatchNorm running mean/var
          2. Re-enable each Dropout layer → keeps stochastic channel zeroing

        After calling this, run N stochastic forward passes with torch.no_grad():
          model.enable_mc_dropout()
          with torch.no_grad():
              preds = [F.softmax(model(x), dim=-1) for _ in range(50)]
          mean = torch.stack(preds).mean(0)   # [B, 6] — best estimate
          std  = torch.stack(preds).std(0)    # [B, 6] — uncertainty per class
        """
        self.eval()                           # freeze BN
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()                     # re-enable Dropout only
        logger.debug("MC Dropout enabled — BN frozen, Dropout active")

    # ── Fine-tuning helpers ────────────────────────────────────────────────

    def freeze_cnn(self) -> None:
        """
        Freeze all CNN block parameters (requires_grad = False).

        Called at the start of Phase 2 (TESS fine-tuning) when
        CFG.phase2_freeze_cnn = True. CNN features learned on Kepler data
        transfer well to TESS — only the LSTM, attention, and head need
        adapting to the TESS noise floor and class distribution.

        The optimiser in train.py must be re-initialised after calling this
        so that frozen parameters are excluded from its parameter groups.
        """
        n_frozen = 0
        for block in (self.cnn1, self.cnn2, self.cnn3):
            for param in block.parameters():
                param.requires_grad = False
                n_frozen += param.numel()
        logger.info(
            f"CNN blocks frozen — {n_frozen:,} parameters excluded from gradient"
        )

    def unfreeze_all(self) -> None:
        """
        Unfreeze all parameters (requires_grad = True).

        Called at the start of Hyderabad fine-tuning when
        CFG.hyderabad_freeze_cnn = False (full fine-tune on ISRO data).
        """
        for param in self.parameters():
            param.requires_grad = True
        logger.info(
            f"All parameters unfrozen — {self.count_parameters():,} trainable"
        )

    # ── Parameter counting ─────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Total parameters including frozen."""
        return sum(p.numel() for p in self.parameters())

    def trainable_ratio(self) -> float:
        """Fraction of parameters currently trainable (useful after freeze_cnn)."""
        total    = self.count_all_parameters()
        trainable = self.count_parameters()
        return trainable / total if total > 0 else 0.0

    # ── Shape verification ─────────────────────────────────────────────────

    def verify_shapes(self, T: int = CFG.input_length) -> None:
        """
        Run a dummy forward pass and print tensor shapes at every stage.
        Call after building the model to confirm the architecture is correct.

        Args:
            T : temporal length to test with (default 200 from CFG)
        """
        self.eval()
        with torch.no_grad():
            x = torch.randn(2, 1, T)

            print(f"\n{'='*60}")
            print(f"  TransitClassifier — shape verification  (T={T})")
            print(f"{'='*60}")
            print(f"  Input              : {list(x.shape)}")

            x1 = self.cnn1(x)
            print(f"  After cnn1         : {list(x1.shape)}")

            x2 = self.cnn2(x1)
            print(f"  After cnn2         : {list(x2.shape)}")

            x3 = self.cnn3(x2)
            print(f"  After cnn3         : {list(x3.shape)}")

            xp = x3.permute(0, 2, 1).contiguous()
            print(f"  After permute      : {list(xp.shape)}")

            xl1 = self.lstm1(xp)
            print(f"  After lstm1        : {list(xl1.shape)}")

            xl2 = self.lstm2(xl1)
            print(f"  After lstm2        : {list(xl2.shape)}")

            xa = self.attn(xl2)
            print(f"  After attention    : {list(xa.shape)}")
            print(f"  Attention weights  : {list(self.attn.last_weights.shape)}")

            xout = self.head(xa)
            print(f"  After head (logits): {list(xout.shape)}")
            print(f"{'='*60}")
            print(f"  Total parameters   : {self.count_all_parameters():,}")
            print(f"  Trainable params   : {self.count_parameters():,}")
            print(f"  Trainable ratio    : {self.trainable_ratio():.1%}")
            print(f"{'='*60}")

            # Critical shape assertions
            assert xout.shape == (2, self._num_classes), (
                f"Output shape {list(xout.shape)} != expected [2, {self._num_classes}]"
            )
            assert self.attn.last_weights.shape == (2, xp.shape[1]), (
                f"Attention weights shape {list(self.attn.last_weights.shape)} "
                f"!= expected [2, {xp.shape[1]}]"
            )
            print("  ✅ Shape verification passed\n")

    def __repr__(self) -> str:
        return (
            f"TransitClassifier(\n"
            f"  cnn    : {self._cnn_channels}\n"
            f"  lstm   : hidden={self._lstm_hidden_sizes}, bidirectional=True\n"
            f"  attn   : dim={self.lstm2.output_dim}\n"
            f"  head   : {self.lstm2.output_dim}→{self._num_classes} classes\n"
            f"  params : {self.count_all_parameters():,} total, "
            f"{self.count_parameters():,} trainable\n"
            f")"
        )
