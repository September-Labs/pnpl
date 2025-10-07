from torch.utils.data import Dataset
import torch
import numpy as np
import time
import h5py
import json
import hashlib
from pathlib import Path
from tqdm import tqdm


class GroupedDataset(Dataset):
    def __init__(self, original_dataset=None, grouped_samples=10, drop_remaining=False, shuffle=False, average_grouped_samples=True,
                 prefetch_labels=True, shuffle_seed=None, 
                 preprocessed_path=None, load_to_memory=False, transform=None):
        """
        Groups n samples from the original dataset by label.
        Can either group from an original dataset or load preprocessed groups.

        Parameters:
        - original_dataset: The original dataset to group (if preprocessing)
        - grouped_samples: The number of samples to group over
        - drop_remaining: Whether to drop the last group if it is incomplete
        - shuffle: Whether to shuffle the samples
        - average_grouped_samples: Whether to average the grouped samples
        - prefetch_labels: Pre-fetch all labels to minimize dataset calls (default: True)
        - shuffle_seed: Random seed for shuffle (for reproducibility)
        - preprocessed_path: Path to preprocessed HDF5 file (for fast loading)
        - load_to_memory: Load entire dataset to memory (for preprocessed data)
        - transform: Optional transform to apply to data
        """

        # Check if loading preprocessed data or creating new groups
        if preprocessed_path is not None:
            self._load_preprocessed(preprocessed_path, load_to_memory)
            self.transform = transform
            return
        
        # Original grouping logic
        if original_dataset is None:
            raise ValueError("Either original_dataset or preprocessed_path must be provided")
            
        if (not drop_remaining and not average_grouped_samples):
            raise ValueError(
                "drop_remaining and average_grouped_samples cannot both be False. Otherwise the dimension of the output will be inconsistent.")

        self.original_dataset = original_dataset
        self.average_grouped_samples = average_grouped_samples
        self.grouped_samples = grouped_samples
        self.prefetch_labels = prefetch_labels
        self.shuffle_seed = shuffle_seed
        self.transform = transform
        
        dataset_size = len(original_dataset)
        if shuffle:
            if shuffle_seed is not None:
                torch.manual_seed(shuffle_seed)
            indices = torch.randperm(dataset_size)
        else:
            indices = torch.arange(dataset_size)
        
        # Pre-fetch all labels if requested for efficient grouping
        if self.prefetch_labels:
            labels = []
            
            # Check if dataset has efficient get_label method
            if hasattr(original_dataset, 'get_label'):
                print("Using fast label fetching...")
                for idx in tqdm(indices, desc="Loading labels"):
                    label = original_dataset.get_label(idx.item())
                    labels.append(label)
            else:
                # Fallback to original implementation
                batch_size = 1000
                print("Fetching labels...")
                for i in tqdm(range(0, dataset_size, batch_size), desc="Loading labels"):
                    batch_indices = indices[i:min(i+batch_size, dataset_size)]
                    batch_labels = [original_dataset[idx.item()][1].item() for idx in batch_indices]
                    labels.extend(batch_labels)
            
            # Group indices by label using vectorized operations
            labels_tensor = torch.tensor(labels)
            unique_labels = torch.unique(labels_tensor)
            
            self.groups = []
            
            print("Grouping samples by label...")
            for label in tqdm(unique_labels, desc="Creating groups"):
                label_mask = labels_tensor == label
                label_indices = indices[label_mask]
                
                # Split into groups of grouped_samples
                for i in range(0, len(label_indices), grouped_samples):
                    group = label_indices[i:i+grouped_samples]
                    if len(group) == grouped_samples or not drop_remaining:
                        self.groups.append(group.tolist())
        else:
            # Original implementation
            self.groups = []
            self.partial_groups = {}
            
            print("Creating groups...")
            for i in tqdm(indices, desc="Grouping samples"):
                label = original_dataset[i.item()][1].item()
                group = self.partial_groups.get(label, [])
                group.append(i.item())
                self.partial_groups[label] = group
                if (len(group) == grouped_samples):
                    self.groups.append(group)
                    self.partial_groups[label] = []

            if not drop_remaining:
                for group in self.partial_groups.values():
                    if group:
                        self.groups.append(group)
        
        # Groups are now ready

    def _load_preprocessed(self, h5_path, load_to_memory):
        """Load preprocessed grouped dataset from HDF5."""
        self.h5_path = Path(h5_path)
        self.load_to_memory = load_to_memory
        
        # Open file to read metadata
        with h5py.File(self.h5_path, 'r') as f:
            self.n_groups = f.attrs['n_groups']
            self.n_channels = f.attrs['n_channels']
            self.n_timepoints = f.attrs['n_timepoints']
            self.grouped_samples = f.attrs['grouped_samples']
            self.average_grouped_samples = f.attrs.get('average_grouped_samples', True)
            
            # Load groups for compatibility
            self.groups = []
            with h5py.File(self.h5_path, 'r') as f:
                if 'group_indices' in f:
                    for i in range(self.n_groups):
                        self.groups.append(f['group_indices'][i][:].tolist())
        
        # Load entire dataset to memory if requested
        if self.load_to_memory:
            print(f"Loading {self.n_groups} groups to memory...")
            with h5py.File(self.h5_path, 'r') as f:
                self.data = torch.from_numpy(f['data'][:])
                self.labels = torch.from_numpy(f['labels'][:])
            print(f"Loaded {self.data.shape[0]} groups, "
                  f"total size: {self.data.nbytes / 1024**3:.2f} GB")
            self.h5_file = None
        else:
            # Keep file handle open for efficient access
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True)
            self.data = self.h5_file['data']
            self.labels = self.h5_file['labels']
        
        # Set original_dataset to None for preprocessed data
        self.original_dataset = None

    def __len__(self):
        return len(self.groups) if hasattr(self, 'groups') else self.n_groups

    def __getitem__(self, idx):
        # Check if using preprocessed data
        if hasattr(self, 'h5_path'):
            if idx >= self.n_groups:
                raise IndexError(f"Index {idx} out of range for dataset with {self.n_groups} groups")
            
            # Load from preprocessed data
            if self.load_to_memory:
                data = self.data[idx]
                label = self.labels[idx]
            else:
                data = torch.from_numpy(self.data[idx])
                label = self.labels[idx].item()
            
            # Apply transform if specified
            if self.transform is not None:
                data = self.transform(data)
            
            return data, label
        
        # Original implementation for on-the-fly grouping
        group = self.groups[idx]
        
        # Load samples from the group
        samples = [self.original_dataset[i] for i in group]
        samples_data = [sample[0] for sample in samples]
        
        if self.average_grouped_samples:
            data = torch.stack(samples_data)
            data = data.mean(dim=0)
        else:
            data = torch.concat(samples_data, dim=0)
        
        label = samples[0][1]
        return data, label
    
    def __del__(self):
        # Close HDF5 file when object is destroyed
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
    
    @staticmethod
    def preprocess_and_save(original_dataset, output_path, grouped_samples=100,
                           drop_remaining=False, shuffle=True, average_grouped_samples=True,
                           shuffle_seed=None, batch_size=16, num_workers=None):
        """Preprocess dataset and save to HDF5 for fast loading."""
        from tqdm import tqdm
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
        
        # Create grouped dataset
        print("Creating grouped dataset...")
        grouped_dataset = GroupedDataset(
            original_dataset,
            grouped_samples=grouped_samples,
            drop_remaining=drop_remaining,
            shuffle=shuffle,
            average_grouped_samples=average_grouped_samples,
            shuffle_seed=shuffle_seed,
            prefetch_labels=True  # Use efficient grouping with progress bars
        )
        
        # Get dimensions from first sample
        sample_data, sample_label = grouped_dataset[0]
        n_channels = sample_data.shape[0]
        n_timepoints = sample_data.shape[1]
        n_groups = len(grouped_dataset)
        
        # Create output directory
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Preprocessing {n_groups} groups to {output_path}")
        
        # Determine number of workers
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 40)  # Cap at 40 workers
        print(f"Using {num_workers} workers for parallel processing")
        
        # Create HDF5 file with optimized settings
        with h5py.File(output_path, 'w') as f:
            # Create datasets with chunking optimized for sequential access
            chunk_size = min(32, n_groups)  # 32 groups per chunk
            
            # Main data array
            data_ds = f.create_dataset(
                'data',
                shape=(n_groups, n_channels, n_timepoints),
                dtype='float32',
                chunks=(chunk_size, n_channels, n_timepoints),
                compression=None  # Disable compression for faster writes
            )
            
            # Labels array
            labels_ds = f.create_dataset(
                'labels',
                shape=(n_groups,),
                dtype='i8',
                chunks=(min(n_groups, chunk_size * 16),),
                compression=None  # Disable compression for faster writes
            )
            
            # Store metadata
            f.attrs['n_groups'] = n_groups
            f.attrs['n_channels'] = n_channels
            f.attrs['n_timepoints'] = n_timepoints
            f.attrs['grouped_samples'] = grouped_samples
            f.attrs['average_grouped_samples'] = average_grouped_samples
            f.attrs['drop_remaining'] = drop_remaining
            f.attrs['shuffle'] = shuffle
            f.attrs['shuffle_seed'] = shuffle_seed if shuffle_seed is not None else -1
            f.attrs['created_time'] = time.time()
            
            # Store group indices for reference
            dt = h5py.special_dtype(vlen=np.int32)
            indices_ds = f.create_dataset('group_indices', shape=(n_groups,), dtype=dt)
            
            # Process and save all groups
            print(f"Processing {n_groups} groups...")
            
            # Check if we can optimize for LibriBrain datasets
            if hasattr(original_dataset, 'samples') and hasattr(original_dataset, 'open_h5_datasets'):
                print("Optimizing for LibriBrain dataset...")
                
                # Preload all unique runs' data into memory
                print("Preloading MEG data into memory...")
                run_data_cache = {}
                unique_runs = set()
                
                # Find all unique runs
                for group in grouped_dataset.groups:
                    for sample_idx in group:
                        sample = original_dataset.samples[sample_idx]
                        run_key = (sample[0], sample[1], sample[2], sample[3])
                        unique_runs.add(run_key)
                
                # Load all run data into memory
                print(f"Loading {len(unique_runs)} unique runs into memory...")
                for run_key in tqdm(unique_runs, desc="Loading runs"):
                    if run_key not in original_dataset.open_h5_datasets:
                        h5_path = original_dataset._ids_to_h5_path(*run_key)
                        h5_file = original_dataset._open_h5_file(h5_path)
                        h5_dataset = h5_file["data"]
                        original_dataset.open_h5_datasets[run_key] = h5_dataset
                    
                    # Load entire run into memory
                    run_data_cache[run_key] = np.array(original_dataset.open_h5_datasets[run_key])
                
                print(f"Loaded {sum(data.nbytes for data in run_data_cache.values()) / 1024**3:.2f} GB into memory")
                
                # Now process groups using cached data
                with tqdm(total=n_groups, desc="Processing groups") as pbar:
                    for group_idx, group in enumerate(grouped_dataset.groups):
                        samples_data = []
                        label = None
                        
                        for sample_idx in group:
                            sample = original_dataset.samples[sample_idx]
                            run_key = (sample[0], sample[1], sample[2], sample[3])
                            onset = sample[4]
                            
                            # Get data from cache
                            h5_data = run_data_cache[run_key]
                            start = max(0, int((onset + original_dataset.tmin) * original_dataset.sfreq))
                            end = start + original_dataset.points_per_sample
                            data = h5_data[:, start:end].copy()
                            
                            # Standardize if needed
                            if original_dataset.standardize:
                                data = (data - original_dataset.channel_means[:, np.newaxis]) / original_dataset.channel_stds[:, np.newaxis]
                            
                            # Clip if needed
                            if original_dataset.clipping_boundary is not None:
                                data = np.clip(data, -original_dataset.clipping_boundary, original_dataset.clipping_boundary)
                            
                            samples_data.append(data)
                            
                            # Get label (only need it once)
                            if label is None:
                                if hasattr(original_dataset, 'get_label'):
                                    label = original_dataset.get_label(sample_idx)
                                else:
                                    label = 0  # fallback
                        
                        # Average or concatenate samples
                        if average_grouped_samples:
                            data = np.stack(samples_data).mean(axis=0)
                        else:
                            data = np.concatenate(samples_data, axis=0)
                        
                        # Write to HDF5
                        data_ds[group_idx] = data
                        labels_ds[group_idx] = label
                        indices_ds[group_idx] = np.array(group, dtype=np.int32)
                        
                        pbar.update(1)
            else:
                # Fallback to original implementation
                with tqdm(total=n_groups) as pbar:
                    for idx in range(n_groups):
                        data, label = grouped_dataset[idx]
                        data_ds[idx] = data.numpy()
                        labels_ds[idx] = label.item() if torch.is_tensor(label) else label
                        indices_ds[idx] = np.array(grouped_dataset.groups[idx], dtype=np.int32)
                        pbar.update(1)
        
        print(f"Saved {n_groups} groups to {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024**3:.2f} GB")
        
        return output_path


if __name__ == "__main__":
    # data_path = "/data/engs-pnpl/datasets/Sherlock1/derivatives/preproc"
    data_path = "/Users/mirgan/Sherlock1/derivatives/serialized/default"
    events_path = "/Users/mirgan/Sherlock1/"
    """data_path = "/data/engs-pnpl/datasets/Sherlock1/derivatives/serialized/default"
    events_path = "/data/engs-pnpl/datasets/Sherlock1/" """
    preprocessing_name = "bads+headpos+sss+notch+bp+ds"
    include_subjects = ['0']
    from pnpl.datasets.parkerjones2025.dataset import ParkerJones2025
    train_data = ParkerJones2025(
        data_path, preprocessing_name=preprocessing_name, include_subjects=include_subjects, events_path=events_path,
        include_runs=["1"],
        include_sessions=["1"],
        include_tasks=["Sherlock1"],
        standardize=True,
        clipping_factor=10,
    )
    grouped_data = GroupedDataset(
        train_data, grouped_samples=10, average_grouped_samples=False)
    print(len(train_data))
    print(len(grouped_data))
    print(grouped_data[0])
    print(grouped_data[1])
