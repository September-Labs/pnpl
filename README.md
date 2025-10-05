# PNPL - MEG Data Loading Library

A high-performance Python library for loading and preprocessing MEG (Magnetoencephalography) data, with specialized support for the LibriBrain 2025 competition datasets.

## Key Features

### Preprocessing for Fast Data Loading
The main contribution of this library is the ability to preprocess and group samples in advance, resulting in much faster data loading during model training. Instead of grouping samples in real-time (which can be slow), you can preprocess your data once and then load it quickly during training.

```bash
# Step 1: Preprocess data once (groups samples and saves to HDF5)
python scripts/preprocess_libribrain.py --data-path /path/to/libribrain --grouped-samples 100

# Step 2: Load preprocessed data in your training code (much faster!)
from pnpl.datasets import GroupedDataset
train_dataset = GroupedDataset(preprocessed_path="./preprocessed_data/train_grouped.h5")
```

The preprocessing script supports configurable group sizes (e.g., 10, 30, 100 samples) and parallel processing for efficiency.

### Additional Features

- **LibriBrain competition support**: Built-in support for LibriBrain 2025 phoneme and speech datasets
- **Flexible data handling**: Support for various preprocessing configurations and standardization options
- **Memory-efficient loading**: Options for memory mapping and loading to memory
- **Hugging Face integration**: Automatic dataset downloading from Hugging Face Hub

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/September-Labs/pnpl.git
```

### Install with preprocessing support

```bash
pip install "pnpl[preprocessing] @ git+https://github.com/September-Labs/pnpl.git"
```

### Development installation

```bash
git clone https://github.com/September-Labs/pnpl.git
cd pnpl
pip install -e ".[dev]"
```

## Quick Start

### Preprocessing Workflow (Recommended)

For the best performance, preprocess your LibriBrain data once, then use the preprocessed files for training:

```bash
# 1. Preprocess all partitions with 100-sample grouping
python scripts/preprocess_libribrain.py \
    --data-path /path/to/libribrain \
    --grouped-samples 100 \
    --output-dir ./preprocessed_data

# 2. Or preprocess specific partitions with custom settings
python scripts/preprocess_libribrain.py \
    --data-path /path/to/libribrain \
    --grouped-samples 30 \
    --partitions train validation \
    --num-workers 8
```

Then in your training code:

```python
from pnpl.datasets import GroupedDataset

# Load preprocessed data (fast!)
train_dataset = GroupedDataset(
    preprocessed_path="./preprocessed_data/train_grouped.h5",
    load_to_memory=True  # Optional: load entire dataset to memory
)

val_dataset = GroupedDataset(
    preprocessed_path="./preprocessed_data/validation_grouped.h5",
    load_to_memory=True
)
```

### Standard Loading (Without Preprocessing)

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

### How to Use Both Workflows

```python
from pnpl.datasets import GroupedDataset, LibriBrainPhoneme

# Option 1: Real-time grouping (slower)
base_dataset = LibriBrainPhoneme(
    data_path="/path/to/libribrain",  # Your LibriBrain data path
    partition="train"
)
grouped_dataset = GroupedDataset(
    original_dataset=base_dataset,
    grouped_samples=100,
    average_grouped_samples=True,
    shuffle=True
)

# Option 2: Load preprocessed data (much faster - recommended!)
grouped_dataset = GroupedDataset(
    preprocessed_path="./preprocessed_data/train_grouped.h5",
    load_to_memory=True
)
```

### Data Path Notes

The LibriBrain dataset can be:
1. **Downloaded automatically** from HuggingFace (stored in HF cache: `~/.cache/huggingface/hub/`)
2. **Provided locally** at a path you specify with `--data-path`

When using the preprocessing script, provide the root path where your LibriBrain data is stored (or where you want it to be downloaded).

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
  url = {https://github.com/September-Labs/pnpl}
}
```

## Acknowledgments

This library was developed for the LibriBrain 2025 competition. Special thanks to the competition organizers and the neuroimaging community.