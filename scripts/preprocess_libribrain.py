#!/usr/bin/env python3
"""
Preprocess LibriBrain phoneme datasets for fast loading.

This script groups samples in advance and saves them to HDF5 files,
resulting in much faster data loading during training.
"""

import argparse
import time
from pathlib import Path
import sys

# Add parent directory to path to import pnpl
sys.path.append(str(Path(__file__).parent.parent))

from pnpl.datasets import LibriBrainPhoneme, GroupedDataset


def preprocess_partition(
    data_path,
    partition,
    output_dir,
    grouped_samples,
    num_workers,
    channel_means=None,
    channel_stds=None
):
    """
    Preprocess a single partition and save to HDF5.

    Args:
        data_path: Path to LibriBrain dataset
        partition: 'train', 'validation', or 'test'
        output_dir: Directory to save preprocessed files
        grouped_samples: Number of samples to group together
        num_workers: Number of parallel workers for processing
        channel_means: Pre-computed channel means (from training set)
        channel_stds: Pre-computed channel stds (from training set)

    Returns:
        Tuple of (channel_means, channel_stds, output_path)
    """
    print(f"\n{'='*60}")
    print(f"Processing {partition} partition")
    print(f"{'='*60}")

    # Create base dataset
    print(f"Loading {partition} dataset...")
    start_time = time.time()

    base_dataset = LibriBrainPhoneme(
        data_path=data_path,
        partition=partition,
        tmin=-0.2,
        tmax=0.6,
        standardize=True,
        channel_means=channel_means,
        channel_stds=channel_stds,
    )

    # Extract standardization params from training set
    if partition == 'train' and channel_means is None:
        channel_means = base_dataset.channel_means
        channel_stds = base_dataset.channel_stds
        print(f"Extracted standardization parameters from training set")

    print(f"Base dataset size: {len(base_dataset)} samples")
    print(f"Time to load: {time.time() - start_time:.2f}s")

    # Determine output path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{partition}_grouped.h5"

    # Preprocess and save
    print(f"Grouping {grouped_samples} samples and saving to {output_path}...")
    start_time = time.time()

    GroupedDataset.preprocess_and_save(
        base_dataset,
        output_path,
        grouped_samples=grouped_samples,
        shuffle=True if partition == 'train' else False,
        average_grouped_samples=True,
        shuffle_seed=42,  # Fixed seed for reproducibility
        batch_size=256,
        num_workers=num_workers
    )

    print(f"Preprocessing completed in {time.time() - start_time:.2f}s")
    print(f"Saved to: {output_path}")

    return channel_means, channel_stds, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LibriBrain phoneme datasets for fast loading"
    )

    # Required arguments
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to LibriBrain dataset (required)"
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./preprocessed_data",
        help="Directory to save preprocessed files (default: ./preprocessed_data)"
    )

    parser.add_argument(
        "--grouped-samples",
        type=int,
        default=100,
        help="Number of samples to group together (default: 100)"
    )

    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=["train", "validation", "test"],
        default=["train", "validation", "test"],
        help="Partitions to preprocess (default: all)"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for processing (default: 4)"
    )

    args = parser.parse_args()

    # Print configuration
    print("Preprocessing Configuration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Grouped samples: {args.grouped_samples}")
    print(f"  Partitions: {args.partitions}")
    print(f"  Number of workers: {args.num_workers}")

    # Track total time
    total_start = time.time()

    # Process partitions
    channel_means = None
    channel_stds = None

    # Always process train first if included (to get standardization params)
    if "train" in args.partitions:
        channel_means, channel_stds, _ = preprocess_partition(
            args.data_path,
            "train",
            args.output_dir,
            args.grouped_samples,
            args.num_workers
        )

    # Process other partitions
    for partition in args.partitions:
        if partition != "train":  # Skip train as we already processed it
            preprocess_partition(
                args.data_path,
                partition,
                args.output_dir,
                args.grouped_samples,
                args.num_workers,
                channel_means=channel_means,
                channel_stds=channel_stds
            )

    # Print summary
    print(f"\n{'='*60}")
    print("Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"Total time: {time.time() - total_start:.2f}s")
    print(f"Output files saved to: {args.output_dir}")

    # Print usage instructions
    print("\nTo use the preprocessed data in your code:")
    print("```python")
    print("from pnpl.datasets import GroupedDataset")
    print(f"train_dataset = GroupedDataset(preprocessed_path='{args.output_dir}/train_grouped.h5')")
    print(f"val_dataset = GroupedDataset(preprocessed_path='{args.output_dir}/validation_grouped.h5')")
    print("```")


if __name__ == "__main__":
    main()