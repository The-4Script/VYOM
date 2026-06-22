"""
classifier/test.py

Single-curve inference for the trained six-class transit classifier.

Accepts:
  --npy  path/to/folded.npy   pre-folded 200-point array (fastest — no FITS needed)
  --fits path/to/star.fits    FITS file, folded on-the-fly using --period and --t0

Inference:
  Monte Carlo Dropout — 50 forward passes with dropout ACTIVE, BN FROZEN.
  Returns per-class mean probability ± std (uncertainty estimate).

Output:
  - Formatted prediction table printed to stdout
  - Plain English classification summary
  - 3-panel matplotlib figure:
      Panel 1  — folded light curve input
      Panel 2  — class probability bar chart with ±1σ MC Dropout error bars
      Panel 3  — mean attention weights over phase bins (25 LSTM positions)
  - Optional JSON output for pipeline integration

Usage:
  python -m classifier.test --npy data/samples/classifier/test/TIC123_cls0_0000.npy
  python -m classifier.test --fits data/raw/tess/tess_lc.fits --period 3.14 --t0 2458325.5
  python -m classifier.test --npy path/to/folded.npy --mc-passes 100
  python -m classifier.test --npy path/to/folded.npy --no-plot
  python -m classifier.test --npy path/to/folded.npy --save-plot results/pred.png
  python -m classifier.test --npy path/to/folded.npy --json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .config import CFG
from .model import TransitClassifier
from .dataset import phase_fold, _read_fits_lc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
})

# Per-class colour — consistent with evaluate.py
CLASS_COLORS = [
    "#2563EB",  # 0 PT  — blue
    "#DC2626",  # 1 EB  — red
    "#D97706",  # 2 BEB — amber
    "#7C3AED",  # 3 HEB — purple
    "#059669",  # 4 SV  — emerald
    "#374151",  # 5 IA  — slate
]

# ─────────────────────────────────────────────────────────────────────────────
# Plain English explanations — rule-based, no LLM
# ─────────────────────────────────────────────────────────────────────────────

# Brief description of each class, shown regardless of confidence level
CLASS_DESCRIPTIONS = {
    0: (
        "Planet Transit (PT)",
        "Periodic dimming consistent with an orbiting planet. "
        "Characterised by a flat-bottomed dip, no secondary eclipse, "
        "and symmetric ingress/egress."
    ),
    1: (
        "Eclipsing Binary (EB)",
        "Two stars orbiting each other. "
        "Typically shows deep symmetric dips; a secondary eclipse near "
        "phase 0.5 is the key distinguishing feature."
    ),
    2: (
        "Background Eclipsing Binary (BEB)",
        "An eclipsing binary in the same photometric aperture as the target. "
        "The contaminating source dilutes the dip depth, making it appear "
        "shallower than it really is."
    ),
    3: (
        "Hierarchical Eclipsing Binary (HEB)",
        "An EB physically bound to and orbiting the target star. "
        "Odd/even transit depth alternation is the primary diagnostic."
    ),
    4: (
        "Stellar Variability (SV)",
        "Signal originates from the star itself — rotation, pulsation, or starspots. "
        "Typically shows a smooth sinusoidal morphology rather than sharp "
        "transit ingress/egress."
    ),
    5: (
        "Instrumental Artifact (IA)",
        "Not astrophysical. Likely correlates with spacecraft momentum dumps, "
        "scattered light, or detector systematics. "
        "Recheck quality flags in the raw FITS file."
    ),
}

# What follow-up observation is recommended for each class
FOLLOWUP_RECOMMENDATIONS = {
    0: "Priority target for radial velocity follow-up to confirm planetary mass.",
    1: "Radial velocity observations will show anti-phase stellar motion.",
    2: "High-resolution imaging (AO / speckle) to resolve the background source.",
    3: "Spectroscopic observation to characterise the bound companion.",
    4: "No planetary follow-up warranted — stellar astrophysics signal.",
    5: "Reprocess with updated quality mask; check raw SAP_FLUX and momentum-dump timing.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    weights_path: Path = CFG.best_weights,
    device:       torch.device = None,
) -> tuple[TransitClassifier, torch.device]:
    """
    Load TransitClassifier from checkpoint.

    Args:
        weights_path : path to trained .pth file
        device       : torch device (auto-detected if None)

    Returns:
        (model, device) — model in eval mode, device used
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Train first: python -m classifier.train"
        )

    model = TransitClassifier().to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    logger.info(
        f"Loaded: {weights_path.name}  "
        f"(epoch={ckpt.get('epoch', '?')}, "
        f"phase={ckpt.get('phase', '?')}, "
        f"best_val_f1={ckpt.get('best_val_f1', '?')})"
    )
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# MC Dropout inference
# ─────────────────────────────────────────────────────────────────────────────

def mc_dropout_predict(
    model:    TransitClassifier,
    x:        torch.Tensor,
    n_passes: int = CFG.mc_dropout_passes,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Monte Carlo Dropout uncertainty estimation for the classifier.

    Activates model.enable_mc_dropout() — this freezes BatchNorm (using
    learned running statistics) but keeps all Dropout layers stochastic.
    Each forward pass uses a different dropout mask → slightly different
    predictions. The spread across passes estimates model uncertainty.

    Why BN must be frozen (eval mode) while Dropout is kept active:
      If BN were in train mode, it would recompute batch statistics from
      just one or a few samples per MC pass, giving wildly unstable outputs.
      Frozen BN uses population statistics from training — stable and correct.

    Args:
        model    : TransitClassifier (loaded, on correct device)
        x        : [1, 1, 200] input tensor — single folded light curve
        n_passes : number of stochastic forward passes (default 50)

    Returns:
        mean_probs : [1, num_classes] — mean softmax probability per class
        std_probs  : [1, num_classes] — std of softmax probabilities per class
        mean_attn  : [1, 25]          — mean attention weights over MC passes
    """
    # enable_mc_dropout: model.eval() freezes BN, then re-enables every Dropout
    model.enable_mc_dropout()

    all_probs: list[torch.Tensor] = []
    all_attn:  list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(x)                          # [1, num_classes]
            probs  = F.softmax(logits, dim=-1)         # [1, num_classes]
            all_probs.append(probs.cpu())

            if model.attn.last_weights is not None:
                all_attn.append(model.attn.last_weights.cpu())  # [1, 25]

    stacked_probs = torch.stack(all_probs, dim=0)   # [n_passes, 1, num_classes]
    mean_probs    = stacked_probs.mean(dim=0)        # [1, num_classes]
    std_probs     = stacked_probs.std(dim=0)         # [1, num_classes]

    if all_attn:
        stacked_attn = torch.stack(all_attn, dim=0)  # [n_passes, 1, 25]
        mean_attn    = stacked_attn.mean(dim=0)       # [1, 25]
    else:
        mean_attn = torch.zeros(1, 25)

    # Restore model to full eval mode after MC passes
    model.eval()

    return mean_probs, std_probs, mean_attn


# ─────────────────────────────────────────────────────────────────────────────
# Build result dict
# ─────────────────────────────────────────────────────────────────────────────

def _build_result(
    folded:     np.ndarray,
    mean_probs: torch.Tensor,
    std_probs:  torch.Tensor,
    mean_attn:  torch.Tensor,
    source_desc: str,
) -> dict:
    """
    Assemble inference output into a structured result dict.

    Args:
        folded      : [200] float32 — phase-folded input curve
        mean_probs  : [1, num_classes] — MC Dropout mean probabilities
        std_probs   : [1, num_classes] — MC Dropout std probabilities
        mean_attn   : [1, 25]          — mean attention weights
        source_desc : human-readable description of the input source

    Returns:
        result dict with all predictions and metadata
    """
    mean_np  = mean_probs.squeeze(0).numpy()   # [num_classes]
    std_np   = std_probs.squeeze(0).numpy()    # [num_classes]
    attn_np  = mean_attn.squeeze(0).numpy()    # [25]

    pred_idx      = int(np.argmax(mean_np))
    pred_name     = CFG.get_class_name(pred_idx)
    pred_conf     = float(mean_np[pred_idx])
    pred_std      = float(std_np[pred_idx])
    pred_fp_prob  = float(1.0 - mean_np[0])   # probability that it's NOT a planet

    # Second-best class (for reporting a runner-up)
    sorted_idx = np.argsort(mean_np)[::-1]
    runner_idx  = int(sorted_idx[1]) if len(sorted_idx) > 1 else -1

    per_class = [
        {
            "class_idx":  c,
            "class_name": CFG.get_class_name(c),
            "prob_mean":  round(float(mean_np[c]), 4),
            "prob_std":   round(float(std_np[c]),  4),
        }
        for c in range(CFG.num_classes)
    ]

    return {
        "source":         source_desc,
        "pred_class_idx": pred_idx,
        "pred_class":     pred_name,
        "confidence":     round(pred_conf,    4),
        "uncertainty":    round(pred_std,     4),
        "fp_probability": round(pred_fp_prob, 4),
        "runner_up_idx":  runner_idx,
        "runner_up":      CFG.get_class_name(runner_idx) if runner_idx >= 0 else "N/A",
        "runner_up_prob": round(float(mean_np[runner_idx]), 4) if runner_idx >= 0 else 0.0,
        "per_class":      per_class,
        "attn_weights":   attn_np.tolist(),
        "folded_curve":   folded.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inference from .npy
# ─────────────────────────────────────────────────────────────────────────────

def infer_from_npy(
    npy_path:  Path,
    model:     TransitClassifier,
    device:    torch.device,
    n_passes:  int = CFG.mc_dropout_passes,
) -> dict:
    """
    Run MC Dropout classification on a pre-folded .npy file.

    The .npy file must contain a 1D float32 array of length CFG.fold_points (200).
    These are produced by classifier/dataset.py and cached in data/samples/classifier/.

    Args:
        npy_path  : path to .npy file containing the folded light curve
        model     : loaded TransitClassifier
        device    : torch device
        n_passes  : MC Dropout passes

    Returns:
        result dict (see _build_result)

    Raises:
        FileNotFoundError : npy_path does not exist
        ValueError        : array shape or dtype mismatch
    """
    if not npy_path.exists():
        raise FileNotFoundError(f"NPY file not found: {npy_path}")

    folded = np.load(npy_path).astype(np.float32)

    # Handle both [200] and [1, 200] shapes from different cache formats
    if folded.ndim == 2 and folded.shape[0] == 1:
        folded = folded.squeeze(0)

    if folded.ndim != 1:
        raise ValueError(
            f"Expected 1D array from {npy_path.name}, got shape {folded.shape}"
        )

    if len(folded) != CFG.fold_points:
        raise ValueError(
            f"Expected {CFG.fold_points} points, got {len(folded)}. "
            "Ensure this .npy was produced by the classifier dataset pipeline."
        )

    logger.info(f"Loaded {npy_path.name} — {len(folded)} phase bins")

    # [200] → [1, 200] → [1, 1, 200]
    x = torch.from_numpy(folded).unsqueeze(0).unsqueeze(0).to(device)

    mean_probs, std_probs, mean_attn = mc_dropout_predict(model, x, n_passes)

    return _build_result(
        folded      = folded,
        mean_probs  = mean_probs,
        std_probs   = std_probs,
        mean_attn   = mean_attn,
        source_desc = str(npy_path),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inference from FITS (fold on the fly)
# ─────────────────────────────────────────────────────────────────────────────

def infer_from_fits(
    fits_path: Path,
    period:    float,
    t0:        float,
    model:     TransitClassifier,
    device:    torch.device,
    source:    str = "tess",
    n_passes:  int = CFG.mc_dropout_passes,
) -> dict:
    """
    Read a TESS/Kepler FITS file, phase-fold it, then run MC Dropout inference.

    Steps:
      1. Read FITS → (time, flux) via dataset._read_fits_lc
      2. Phase-fold on the given period and t0 → 200-point binned segment
      3. Run mc_dropout_predict on the folded array

    Args:
        fits_path : path to TESS or Kepler FITS file
        period    : orbital period in days (from TLS or catalog)
        t0        : reference transit epoch (BJD / BTJD)
        model     : loaded TransitClassifier
        device    : torch device
        source    : "tess" or "kepler" — determines FITS column names
        n_passes  : MC Dropout passes

    Returns:
        result dict (see _build_result)

    Raises:
        FileNotFoundError : fits_path does not exist
        ValueError        : FITS unreadable or folding produces None
    """
    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    logger.info(f"Reading FITS: {fits_path.name}")
    lc = _read_fits_lc(fits_path, source=source)
    if lc is None:
        raise ValueError(
            f"Could not read {fits_path.name} — too few valid points or "
            "FITS format unrecognised. Check quality mask and file integrity."
        )

    time_arr, flux_arr = lc
    logger.info(f"  {len(flux_arr)} valid time steps after quality masking")

    logger.info(f"Phase folding on period={period:.4f} d, t0={t0:.4f}")
    folded = phase_fold(
        time   = time_arr,
        flux   = flux_arr,
        period = period,
        t0     = t0,
        n_bins = CFG.fold_points,
    )
    if folded is None:
        raise ValueError(
            f"Phase folding failed for period={period}, t0={t0}.\n"
            "Ensure the period and t0 are correct for this light curve."
        )

    logger.info(f"Folded to {len(folded)} phase bins")

    x = torch.from_numpy(folded).unsqueeze(0).unsqueeze(0).to(device)

    mean_probs, std_probs, mean_attn = mc_dropout_predict(model, x, n_passes)

    return _build_result(
        folded      = folded,
        mean_probs  = mean_probs,
        std_probs   = std_probs,
        mean_attn   = mean_attn,
        source_desc = f"{fits_path.name}  period={period:.4f}d  t0={t0:.4f}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Formatted output
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result: dict) -> None:
    """
    Print a clean prediction summary to stdout.

    Layout:
      Top block    — predicted class, confidence, uncertainty, runner-up
      Per-class    — probability mean ± std for every class
      Explanation  — plain English description + follow-up recommendation
    """
    pred_idx  = result["pred_class_idx"]
    pred_name = result["pred_class"]
    conf      = result["confidence"]
    unc       = result["uncertainty"]
    fp_prob   = result["fp_probability"]

    print()
    print("=" * 60)
    print("  VYOM CLASSIFIER — PREDICTION")
    print("=" * 60)
    print(f"  Source        : {result['source']}")
    print(f"  Prediction    : {CFG.get_class_name(pred_idx, full=True)}")
    print(f"  Confidence    : {conf * 100:.1f}% ± {unc * 100:.1f}%  (MC Dropout ±1σ)")
    print(f"  Runner-up     : {CFG.get_class_name(result['runner_up_idx'], full=True)} "
          f"({result['runner_up_prob'] * 100:.1f}%)")
    print(f"  FP probability: {fp_prob * 100:.1f}%  (prob that signal is not a PT)")
    print()

    # Per-class breakdown
    print(f"  {'Class':<5} {'Full name':<34} {'Mean':>7} {'±Std':>7}")
    print("  " + "-" * 56)
    for pc in result["per_class"]:
        marker = " ◄" if pc["class_idx"] == pred_idx else ""
        print(
            f"  [{pc['class_idx']}]  "
            f"{CFG.get_class_name(pc['class_idx'], full=True):<34} "
            f"{pc['prob_mean'] * 100:>6.2f}%"
            f"{pc['prob_std']  * 100:>6.2f}%"
            f"{marker}"
        )

    print()

    # Plain English explanation
    full_name, description = CLASS_DESCRIPTIONS[pred_idx]
    followup               = FOLLOWUP_RECOMMENDATIONS[pred_idx]

    print("  CLASSIFICATION SUMMARY")
    print("  " + "-" * 56)
    print(f"  {full_name}  ({conf * 100:.1f}% confidence)")
    print()
    # Wrap description to ~55 chars
    words = description.split()
    line  = "  "
    for word in words:
        if len(line) + len(word) + 1 > 58:
            print(line)
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)
    print()
    print(f"  Recommended follow-up:")
    print(f"    {followup}")

    if conf < 0.5:
        runner_up = CFG.get_class_name(result["runner_up_idx"], full=True)
        print()
        print(
            f"  ⚠️  Low confidence ({conf * 100:.1f}%). "
            f"Consider {runner_up} ({result['runner_up_prob'] * 100:.1f}%) "
            "as an alternative."
        )

    print("=" * 60)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Plot result — 3-panel figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_result(
    result:    dict,
    save_path: Optional[Path] = None,
) -> None:
    """
    3-panel figure:
      Panel 1 — phase-folded input light curve (phase 0→1)
      Panel 2 — per-class probability bar chart with ±1σ MC Dropout error bars
      Panel 3 — mean attention weights over the 25 LSTM phase positions

    The attention panel shows which phase bins the model focused on.
    High attention near phase 0.5 (transit centre) is expected for PT signals.
    High attention spread or at phase 0.5 ± 0.5 may indicate secondary eclipses (EB).

    Args:
        result    : dict returned by infer_from_npy or infer_from_fits
        save_path : if given, save to this path; otherwise plt.show()
    """
    folded      = np.array(result["folded_curve"], dtype=np.float32)
    attn        = np.array(result["attn_weights"],  dtype=np.float32)
    per_class   = result["per_class"]
    pred_idx    = result["pred_class_idx"]

    class_names = [pc["class_name"]  for pc in per_class]
    means       = [pc["prob_mean"]   for pc in per_class]
    stds        = [pc["prob_std"]    for pc in per_class]
    bar_colors  = [
        CLASS_COLORS[pc["class_idx"]]
        if pc["class_idx"] == pred_idx
        else "#CBD5E1"   # muted for non-predicted classes
        for pc in per_class
    ]

    phase_bins = np.linspace(0, 1, len(folded))     # [200] phase axis
    attn_phase = np.linspace(0, 1, len(attn))       # [25] LSTM phase positions

    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(3, 1, hspace=0.48, figure=fig)

    # ── Panel 1: folded light curve ───────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(phase_bins, folded, color="#475569", linewidth=1.0, alpha=0.9)
    ax0.axvline(0.5, color="#94A3B8", linestyle="--", linewidth=0.8, alpha=0.6,
                label="Phase 0.5 (transit centre)")
    ax0.set_xlabel("Phase",              fontsize=10)
    ax0.set_ylabel("Normalised flux",    fontsize=10)
    ax0.set_xlim([-0.01, 1.01])
    ax0.set_title(
        f"Input — Phase-folded light curve  ({len(folded)} bins)\n"
        f"{result['source']}",
        fontsize=9,
    )
    ax0.legend(fontsize=8, loc="upper right")

    # ── Panel 2: probability bars ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    x_pos = np.arange(CFG.num_classes)
    bars  = ax1.bar(
        x_pos, means,
        color   = bar_colors,
        width   = 0.55,
        zorder  = 3,
        alpha   = 0.9,
    )
    ax1.errorbar(
        x_pos, means,
        yerr    = stds,
        fmt     = "none",
        color   = "#1E293B",
        capsize = 4,
        linewidth = 1.4,
        capthick  = 1.4,
        zorder  = 4,
    )

    # Annotate bars with value
    for bar, mean, std in zip(bars, means, stds):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.015,
            f"{mean * 100:.1f}%",
            ha="center", va="bottom",
            fontsize=7.5, color="#1E293B",
        )

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(class_names, fontsize=10)
    ax1.set_ylabel("Probability", fontsize=10)
    ax1.set_ylim([0.0, min(1.15, max(means) + max(stds) + 0.15)])
    ax1.set_title(
        f"Classification — {CFG.get_class_name(pred_idx, full=True)}  "
        f"({result['confidence'] * 100:.1f}% ± {result['uncertainty'] * 100:.1f}%)  "
        f"[MC Dropout, {CFG.mc_dropout_passes} passes]",
        fontsize=9,
    )

    # Highlight predicted class label
    ax1.get_xticklabels()[pred_idx].set_fontweight("bold")
    ax1.get_xticklabels()[pred_idx].set_color(CLASS_COLORS[pred_idx])

    # ── Panel 3: attention weights ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    ax2.fill_between(
        attn_phase, attn,
        color="#7C3AED", alpha=0.35,
    )
    ax2.plot(
        attn_phase, attn,
        color="#7C3AED", linewidth=1.8,
        label="Mean attention weight",
    )
    # Mark the transit centre reference
    ax2.axvline(0.5, color="#94A3B8", linestyle="--", linewidth=0.8, alpha=0.6)

    ax2.set_xlabel("Phase  (25 LSTM positions from CNN)",  fontsize=10)
    ax2.set_ylabel("Attention weight",                      fontsize=10)
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim(bottom=0)
    ax2.set_title(
        "Model attention — phase bins the classifier focused on\n"
        "(high weight near 0.5 = transit-centre focused; "
        "spread = broader morphology matters)",
        fontsize=9,
    )
    ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Vyom Classifier — {CFG.get_class_name(pred_idx, full=True)}",
        fontsize=12, fontweight="bold", y=0.995,
    )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Plot saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — called by pipeline/run_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

def run_classifier(
    folded_npy:   Optional[Path] = None,
    fits_path:    Optional[Path] = None,
    period:       Optional[float] = None,
    t0:           Optional[float] = None,
    weights_path: Path  = CFG.best_weights,
    mc_passes:    int   = CFG.mc_dropout_passes,
    source:       str   = "tess",
) -> dict:
    """
    Minimal public API for pipeline integration.

    Called by pipeline/run_pipeline.py. Accepts either a pre-folded .npy path
    OR a FITS path + period + t0 for on-the-fly folding. Does NOT plot —
    the pipeline handles visualisation separately.

    Args:
        folded_npy   : path to pre-folded 200-point .npy file
        fits_path    : path to FITS file (if folded_npy is None)
        period       : orbital period in days (required if fits_path given)
        t0           : transit epoch in BJD (required if fits_path given)
        weights_path : classifier weights to use
        mc_passes    : MC Dropout forward passes
        source       : "tess" or "kepler" (affects FITS column reading)

    Returns:
        result dict with pred_class, confidence, uncertainty, per_class list

    Raises:
        ValueError : neither folded_npy nor fits_path provided
    """
    model, device = load_model(weights_path)

    if folded_npy is not None:
        return infer_from_npy(folded_npy, model, device, n_passes=mc_passes)

    if fits_path is not None:
        if period is None or t0 is None:
            raise ValueError(
                "period and t0 are required when using --fits mode. "
                "Run TLS detection first to get these values."
            )
        return infer_from_fits(
            fits_path, period, t0, model, device,
            source=source, n_passes=mc_passes,
        )

    raise ValueError(
        "Provide either folded_npy (pre-folded .npy) or "
        "fits_path + period + t0 (fold on the fly)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Vyom six-class transit classifier on a single light curve"
    )

    # Input — mutually exclusive
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument(
        "--npy",  type=str,
        help="Path to pre-folded .npy file (200-point phase-folded array)",
    )
    inp.add_argument(
        "--fits", type=str,
        help="Path to TESS/Kepler FITS file (folds on the fly, requires --period and --t0)",
    )

    # FITS mode parameters
    p.add_argument(
        "--period", type=float, default=None,
        help="Orbital period in days (required with --fits)",
    )
    p.add_argument(
        "--t0", type=float, default=None,
        help="Reference transit epoch in BJD/BTJD (required with --fits)",
    )
    p.add_argument(
        "--fits-source", type=str, default="tess", choices=["tess", "kepler"],
        help="Mission for FITS column name resolution  (default: tess)",
    )

    # Inference settings
    p.add_argument(
        "--weights", type=str, default=str(CFG.best_weights),
        help="Path to trained .pth file  (default: weights/classifier_best.pth)",
    )
    p.add_argument(
        "--mc-passes", type=int, default=CFG.mc_dropout_passes,
        help=f"MC Dropout forward passes  (default: {CFG.mc_dropout_passes})",
    )

    # Output settings
    p.add_argument(
        "--no-plot",   action="store_true",
        help="Skip the 3-panel figure — just print prediction to stdout",
    )
    p.add_argument(
        "--save-plot", type=str, default=None,
        help="Save figure to this path instead of displaying it",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Print full result as JSON to stdout (for pipeline integration)",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    model, device = load_model(Path(args.weights))

    # ── Input resolution ──────────────────────────────────────────────────
    if args.npy:
        result = infer_from_npy(
            npy_path = Path(args.npy),
            model    = model,
            device   = device,
            n_passes = args.mc_passes,
        )

    else:   # --fits mode
        if args.period is None or args.t0 is None:
            raise SystemExit(
                "ERROR: --fits mode requires both --period and --t0.\n"
                "Example: --fits star.fits --period 3.14159 --t0 2458325.5\n"
                "Run TLS/BLS detection first to obtain these values."
            )
        result = infer_from_fits(
            fits_path = Path(args.fits),
            period    = args.period,
            t0        = args.t0,
            model     = model,
            device    = device,
            source    = args.fits_source,
            n_passes  = args.mc_passes,
        )

    # ── Print result ──────────────────────────────────────────────────────
    print_result(result)

    # ── JSON output (for pipeline / scripting) ────────────────────────────
    if args.json:
        # folded_curve and attn_weights are lists — JSON-serialisable already
        safe = {k: v for k, v in result.items()}
        print(json.dumps(safe, indent=2))

    # ── Plot ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        save_path = Path(args.save_plot) if args.save_plot else None
        plot_result(result, save_path=save_path)
