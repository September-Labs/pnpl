"""
PNPL - MEG Data Loading Library

A high-performance library for loading and preprocessing MEG data
with support for the LibriBrain 2025 competition datasets.
"""

__version__ = "0.0.7"
__author__ = "PNPL"
__license__ = "MIT"

# Import main classes for easier access
from .datasets import (
    LibriBrainPhoneme,
    LibriBrainSpeech,
    GroupedDataset,
    LibriBrainCompetitionHoldout
)

from .datasets.hdf5.dataset import HDF5Dataset
from .datasets.libribrain2025.preprocess import fif2h5

__all__ = [
    "LibriBrainPhoneme",
    "LibriBrainSpeech",
    "GroupedDataset",
    "LibriBrainCompetitionHoldout",
    "HDF5Dataset",
    "fif2h5",
    "__version__",
]