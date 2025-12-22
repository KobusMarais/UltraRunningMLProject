"""
Setup script for Ultra-Marathon Pace Prediction Project.

This script makes the project installable as a Python package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ultra-marathon-prediction",
    version="0.1.0",
    author="Ultra Marathon Team",
    author_email="team@example.com",
    description="Machine learning pipeline for ultramarathon pace prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ultra-marathon-prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "scikit-learn>=0.22.0",
        "lightgbm>=2.3.0",
        "matplotlib>=3.0.0",
        "seaborn>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ultra-pipeline=src.pipeline:run_pipeline",
        ],
    },
)
