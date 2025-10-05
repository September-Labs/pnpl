from setuptools import setup, find_packages
import os

# Rename the package directory for proper installation
setup(
    name="pnpl",
    version="0.0.7",
    description="MEG data loading library with preprocessed HDF5 data loading, grouping, and LibriBrain competition support",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    package_dir={"pnpl": "."},
    packages=["pnpl"] + ["pnpl." + pkg for pkg in find_packages()],
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "h5py>=3.0.0",
        "pandas>=2.0.0",
        "mne-bids>=0.14.0",
        "huggingface-hub>=0.20.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "preprocessing": ["mne>=1.0.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    python_requires=">=3.8",
    author="PNPL",
    author_email="",
    url="https://github.com/yourusername/pnpl",  # Update with actual URL
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="MEG, EEG, neuroimaging, brain, LibriBrain, HDF5, preprocessing",
)