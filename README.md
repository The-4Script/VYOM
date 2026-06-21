# Vyom (व्योम)

AI-enabled detection of exoplanets from noisy astronomical light curves.

Built for **ISRO Hackathon 2025 — Problem Statement PS7**.

---

## The problem

NASA's TESS telescope measures the brightness of stars over time. When a planet passes in front of its star, it blocks a tiny sliver of light — sometimes less than 0.1% — creating a small dip called a **transit**. Real planets, especially small or faint ones, often get buried in noise: stellar flicker, spacecraft artifacts, cosmic ray hits. Many existing detection pipelines go straight from lightly-cleaned data to detection, missing planets that a properly denoised signal would reveal.

**Vyom's approach:** clean the signal first, then detect — instead of detecting through the noise.

---

## Pipeline overview

```
Raw TESS data
    ↓
Stage 0 — Preprocessing      clean, normalize, detrend
    ↓
Stage 1 — Denoising          Noise2Noise U-Net removes residual noise
    ↓
Stage 2 — Transit Detection  TLS search for periodic dips
    ↓
Stage 2.5 — Shape Fitting    measure depth, duration, symmetry of the dip
    ↓
Stage 3 — Classification     CNN-LSTM, six-class (real planet vs false positives)
    ↓
Stage 4 — Parameter Estimation   planet radius, orbital period, confidence
    ↓
Offline Dashboard            visual results, no internet required
```

The six classification categories: Planet Transit (PT), Eclipsing Binary (EB), Background Eclipsing Binary (BEB), Hierarchical Eclipsing Binary (HEB), Stellar Variability (SV), Instrumental Artifact (IA).

---

## Tech stack

- **Models**: PyTorch (1D U-Net denoiser, CNN-LSTM classifier)
- **Detection**: Transit Least Squares (TLS), with Box Least Squares (BLS) as fallback
- **Detrending**: `wotan`
- **Data access**: `lightkurve` (TESS/Kepler light curves), TOI/KOI catalogs
- **Dashboard**: Streamlit (fully offline)

---

## Folder structure

```
vyom/
│
├── denoiser/      Model 1 — Noise2Noise U-Net
├── classifier/    Model 2 — CNN-LSTM six-class
├── pipeline/       glue code, all stages chained
├── dashboard/      Streamlit local UI
├── data/           datasets (not committed to git)
├── weights/        trained model weights (not committed to git)
├── results/        plots, metrics, completeness map
├── notebooks/       exploration only, not production
├── tests/           unit tests per module
├── docs/            architecture notes, decision write-ups, report
│
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Team

- [Durvesh Thorat](https://www.linkedin.com/in/durvesh-thorat)
- [Kaustubh Bhoir](https://www.linkedin.com/in/kaustubh-bhoir-ce)
- [Nipun Tamore](https://www.linkedin.com/in/nipun-tamore-21ba5b308)
- [Arnav Patil](https://www.linkedin.com/in/arnav-pradip-patil-3b872b358)

---

## Acknowledgments

Built for ISRO Hackathon 2026 (PS7), in partnership with Hack2Skill.
Uses public TESS and Kepler mission data, TOI and KOI catalogs.
