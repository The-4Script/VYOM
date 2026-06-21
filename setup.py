from setuptools import setup, find_packages

setup(
    name="vyom",
    version="0.1.0",
    description="AI pipeline for exoplanet transit detection from TESS light curves",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "astropy>=5.3",
        "lightkurve>=2.4.0",
        "wotan>=1.10",
        "transitleastsquares>=1.0.31",
        "batman-package>=2.4.9",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "plotly>=5.14.0",
        "streamlit>=1.28.0",
        "tqdm>=4.65.0",
    ],
)
