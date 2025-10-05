# PNPL - MEG Data Loading Library

A high-performance Python library for loading and preprocessing MEG (Magnetoencephalography) data, with specialized support for the LibriBrain 2025 competition datasets.

## Features

- **HDF5-based data loading**: Efficient loading of preprocessed MEG data from HDF5 files
- **Dataset grouping and averaging**: Group multiple samples by label with configurable averaging
- **LibriBrain competition support**: Built-in support for LibriBrain 2025 phoneme and speech datasets
- **Preprocessing pipeline**: Convert raw FIF files to optimized HDF5 format
- **Flexible data handling**: Support for various preprocessing configurations and standardization options
- **Memory-efficient loading**: Options for memory mapping and SWMR (Single Writer Multiple Readers) mode
- **Hugging Face integration**: Automatic dataset downloading from Hugging Face Hub

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/yourusername/pnpl.git
```

### Install with preprocessing support

```bash
pip install "pnpl[preprocessing] @ git+https://github.com/yourusername/pnpl.git"
```

### Development installation

```bash
git clone https://github.com/yourusername/pnpl.git
cd pnpl
pip install -e ".[dev]"
```

## Quick Start

### Loading LibriBrain Phoneme Dataset

```python
from pnpl.datasets import LibriBrainPhoneme

# Load training data
dataset = LibriBrainPhoneme(
    data_path="/path/to/data",
    partition="train",
    tmin=-0.2,
    tmax=0.6,
    standardize=True,
    download=True  # Auto-download from Hugging Face
)

# Get a sample
sample = dataset[0]
meg_data = sample['meg']  # Shape: (channels, time)
phoneme = sample['phoneme']  # Phoneme label
```

### Using GroupedDataset for Averaging

```python
from pnpl.datasets import GroupedDataset, LibriBrainPhoneme

# Create base dataset
base_dataset = LibriBrainPhoneme(
    data_path="/path/to/data",
    partition="train"
)

# Group and average 10 samples
grouped_dataset = GroupedDataset(
    original_dataset=base_dataset,
    grouped_samples=10,
    average_grouped_samples=True,
    shuffle=True
)

# Or load preprocessed grouped data
grouped_dataset = GroupedDataset(
    preprocessed_path="/path/to/grouped_data.h5",
    load_to_memory=True
)
```

### Preprocessing FIF Files to HDF5

```python
from pnpl.datasets.libribrain2025.preprocess import fif2h5

# Convert a single FIF file to HDF5
fif2h5(
    fif_file="/path/to/data.fif",
    dtype=np.float32,  # Optional: convert to float32 for smaller files
    output_dir="/path/to/output",
    chunk_size=50,
    compression="gzip",
    compression_opts=4
)
```

### Loading Competition Holdout Data

```python
from pnpl.datasets import LibriBrainCompetitionHoldout

# Load competition test data
holdout_dataset = LibriBrainCompetitionHoldout(
    data_path="/path/to/data",
    partition="test",
    tmin=0.0,
    tmax=0.5
)
```

## Dataset Structure

The library expects data to be organized in the following structure:

```
data_path/
├── train/
│   ├── subject_01/
│   │   ├── run_01.h5
│   │   ├── run_02.h5
│   │   └── ...
│   └── ...
├── validation/
└── test/
```

## API Reference

### Core Classes

- `LibriBrainPhoneme`: Dataset for phoneme classification task
- `LibriBrainSpeech`: Dataset for speech detection task
- `LibriBrainCompetitionHoldout`: Dataset for competition evaluation
- `GroupedDataset`: Dataset wrapper for grouping and averaging samples
- `HDF5Dataset`: Base class for HDF5-based datasets

### Key Parameters

- `data_path`: Root directory for dataset storage
- `partition`: Data split ("train", "validation", "test")
- `tmin`, `tmax`: Time window for MEG data extraction
- `preprocessing_str`: Preprocessing pipeline configuration
- `standardize`: Apply channel-wise standardization
- `download`: Auto-download from Hugging Face Hub

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- h5py >= 3.0.0
- pandas >= 2.0.0
- mne-bids >= 0.14.0
- huggingface-hub >= 0.20.0

Optional for preprocessing:
- mne >= 1.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{pnpl2024,
  title = {PNPL: MEG Data Loading Library},
  year = {2024},
  url = {https://github.com/yourusername/pnpl}
}
```

## Acknowledgments

This library was developed for the LibriBrain 2025 competition. Special thanks to the competition organizers and the neuroimaging community.