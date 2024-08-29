import glob
import os
from tqdm import tqdm
from aegnn.datasets.base.event_dm import EventDataModule

import h5py
import torch
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from typing import List, Optional, Callable
import numpy as np
import pickle

from .utils.normalization import normalize_time

class Ball(EventDataModule):
    def __init__(self, 
                 batch_size: int = 64, 
                 shuffle: bool = True, 
                 num_workers: int = 8, 
                 pin_memory: bool = False, 
                 transform: Optional[Callable[[Data], Data]] = None):
        img_shape = (1280, 720)
        super(Ball, self).__init__(
            img_shape=img_shape,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            transform=transform
        )
        self.save_hyperparameters({
            "preprocessing": {
                "r": 3.0,
                "d_max": 128,
                "n_samples": 25000,
                "sampling": True
            }
        })

    def _prepare_dataset(self, mode: str):
        raw_files = self.raw_files(mode)
        print(f"Found {len(raw_files)} raw files in dataset (mode = {mode})")
        total_bbox_count = len(raw_files)
        print(f"Total count of (filtered) bounding boxes = {total_bbox_count}")

        for rf in tqdm(raw_files):
            self._processing(rf, self.root)

    def _processing(self, rf: str, root: str):
        # Load data from HDF5 using custom loader
        loader = self.HDF5Loader(rf)
        event_x, event_y, event_t = loader.get_position()
        gt_x, gt_y, gt_t = loader.get_bbox_position()

        # Format bounding boxes into (ts, x, y, w, h, class_id) - assuming single class "ball"
        # Width and height are set to 1 for simplicity; adjust if actual size data is available.
        bbox = np.column_stack((gt_t, gt_x, gt_y, np.ones(len(gt_t)), np.ones(len(gt_t)), np.zeros(len(gt_t))))

        # Convert bounding boxes to a tensor
        bbox_tensor = torch.tensor(bbox, dtype=torch.float)

        # Labels (assuming one class: 'ball')
        labels = np.array([0] * len(gt_x))  # Class ID for 'ball'

        # Raw event data (assumed indices and counts)
        raw_start = 0
        raw_num_events = len(event_x)  # Total number of events
        raw_data = (raw_start, raw_num_events)  # (start index, count)

        # Sub-sample events or use all events (based on 'n_samples' parameter)
        num_samples = self.hparams['preprocessing']['n_samples']
        if raw_num_events > num_samples:
            sample_idx = np.random.choice(np.arange(raw_num_events), size=num_samples, replace=False)
        else:
            sample_idx = np.arange(raw_num_events)  # Use all events if fewer than the sample size

        # Prepare positions array including time normalization
        pos = np.column_stack((event_x, event_y, event_t))
        pos = torch.tensor(pos[sample_idx], dtype=torch.float)
        pos[:, 2] = normalize_time(pos[:, 2])

        # Generate graph edges with radius_graph
        edge_index = radius_graph(pos, r=self.hparams['preprocessing']['r'], 
                                max_num_neighbors=self.hparams['preprocessing']['d_max'])

        # Create the sample dictionary for saving
        sample_dict = {
            'bbox': bbox_tensor,  # Bounding boxes
            'label': labels,      # Labels
            'raw_file': rf,       # File path for reference
            'raw': raw_data,      # Raw event data indices
            'sample_idx': sample_idx,  # Sample indices
            'edge_index': edge_index.cpu()  # Edge indices for the graph
        }

        # Save processed data to a pickle file
        processed_dir = os.path.join(root, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        processed_file = os.path.join(processed_dir, os.path.basename(rf).replace(".hdf5", ".pkl"))
        with open(processed_file, 'wb') as f:
            pickle.dump(sample_dict, f)

    def _load_processed_file(self, f_path: str) -> Data:
        with open(f_path, 'rb') as f:
            data_dict = pickle.load(f)

        # Extract features (x), positions (pos), and edges (edge_index)
        sample_idx = data_dict['sample_idx']
        x = torch.tensor(sample_idx, dtype=torch.float32).view(-1, 1)  # Assume x is a single feature for each event

        # Extract positions (x, y, t) and normalize time
        pos = np.column_stack((data_dict['bbox'][:, 1],  # x
                               data_dict['bbox'][:, 2],  # y
                               data_dict['bbox'][:, 0]))  # t (timestamp)
        pos = torch.tensor(pos, dtype=torch.float32)
        pos[:, 2] = normalize_time(pos[:, 2])  # Normalize time

        # Extract bounding boxes and convert to tensor of shape [L, 5] (x, y, w, h, class_id)
        bbox = torch.tensor(data_dict['bbox'][:, 1:6], dtype=torch.long)  # Extract (x, y, w, h, class_id)

        # Labels for bounding boxes
        y = torch.tensor(data_dict['label'], dtype=torch.long)

        # Edge index (2, P) - connections between nodes
        edge_index = data_dict['edge_index'].long()

        # Create PyTorch Geometric Data object
        data = Data(
            x=x,  # Features of events
            pos=pos,  # Positions including time
            edge_index=edge_index,  # Graph structure
            bbox=bbox,  # Bounding box annotations
            y=y,  # Labels corresponding to bounding boxes
            file_id=f_path,  # Optional: Metadata or file identifier
            label=data_dict['label']  # List of labels
        )

        return data

    def raw_files(self, mode: str) -> List[str]:
        return glob.glob(os.path.join(self.root, mode, "*.hdf5"))

    def processed_files(self, mode: str) -> List[str]:
        processed_dir = os.path.join(self.root, "processed")
        return glob.glob(os.path.join(processed_dir, mode, "*.pkl"))

    @property
    def classes(self) -> List[str]:
        return ["ball"]

    class HDF5Loader:
        def __init__(self, file_path):
            self.file_path = file_path
            with h5py.File(self.file_path, 'r') as hdf5:
                self.event_x = hdf5['event_x'][:]
                self.event_y = hdf5['event_y'][:]
                self.event_t = hdf5['event_t'][:]
                self.event_p = hdf5['event_p'][:]

                self.gt_x = hdf5['gt_x'][:]
                self.gt_y = hdf5['gt_y'][:]
                self.gt_t = hdf5['gt_t'][:]

        def get_position(self) -> tuple:
            return self.event_x, self.event_y, self.event_t
        
        def get_bbox_position(self) -> tuple:
            return self.gt_x, self.gt_y, self.gt_t
