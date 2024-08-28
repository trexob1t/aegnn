import glob
import os
from tqdm import tqdm
from aegnn.datasets.base.event_dm import EventDataModule

import h5py
import torch
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from typing import List, Optional
import numpy as np
import pickle

class Ball(EventDataModule):
    def __init__(self, 
                 batch_size: int = 64, 
                 shuffle: bool = True, 
                 num_workers: int = 8, 
                 pin_memory: bool = False, 
                 transform: Optional[Callable[[Data], Data]] = None):
        """
        Initialize the Ball dataset class with HD image size and other parameters.
        
        :param batch_size: The size of batches for the dataloader.
        :param shuffle: Whether to shuffle the data during training.
        :param num_workers: Number of workers for data loading.
        :param pin_memory: Whether to use pinned memory for data loading.
        :param transform: Optional transformation to apply to data samples.
        """
        
        img_shape = (1280, 720)
        
        # Initialize the parent class (EventDataModule) with the appropriate parameters
        super(Ball, self).__init__(
            img_shape=img_shape,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            transform=transform
        )

        # Additional initialization specific to Ball, if needed
        # For example, setting hyperparameters or dataset-specific settings
        self.save_hyperparameters({
            "preprocessing": {
                "r": 3.0,  # Radius for graph construction, adjust as needed
                "d_max": 128,  # Maximum number of neighbors for graph edges
                "n_samples": 25000,  # Number of samples (events) per graph
                "sampling": True  # Whether to subsample events
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
        # Load data using the HDF5Loader
        hdf5_loader = self.HDF5Loader(rf)

        # Retrieve event data and ground truth bounding boxes
        event_x, event_y, event_t = hdf5_loader.get_position()
        gt_x, gt_y, gt_t = hdf5_loader.get_bbox_position()

        # Access hyperparameters
        params = self.hparams['preprocessing']
        radius = params['r']
        max_neighbors = params['d_max']
        num_samples = params['n_samples']
        sampling = params['sampling']

        # Prepare graph data
        pos = torch.tensor(np.vstack((event_x, event_y, event_t)).T, dtype=torch.float32)

        # If subsampling is enabled and there are more events than num_samples, perform subsampling
        if sampling and len(event_x) > num_samples:
            indices = np.random.choice(len(event_x), size=num_samples, replace=False)
            pos = pos[indices]

        # Node features
        x = torch.ones((pos.shape[0], 1), dtype=torch.float32)  # Node features (e.g., ones or other relevant features)

        # Create edges based on spatial proximity using radius graph
        edge_index = radius_graph(pos[:, :2], r=radius, max_num_neighbors=max_neighbors)

        # Prepare bounding box data (assuming gt_x, gt_y are centers and other values can be added as required)
        bbox = torch.tensor(np.vstack((gt_x, gt_y, np.zeros_like(gt_x), np.zeros_like(gt_y), np.zeros_like(gt_x))).T, dtype=torch.long)
        y = bbox[:, -1]  # Assuming the last column is the class_id

        # Labels - creating a list of labels based on the class_id (here a static label is used as a placeholder)
        label = ['ball' for _ in range(bbox.shape[0])]

        # Create Data object
        data = Data(x=x, pos=pos[:, :2], edge_index=edge_index, bbox=bbox, y=y, label=label)

        # Include the file identifier
        data.file_id = rf

        # Define the output path and save the processed file
        processed_dir = os.path.join(root, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        processed_file = os.path.join(processed_dir, os.path.basename(rf).replace(".hdf5", ".pkl"))

        with open(processed_file, 'wb') as f:
            pickle.dump(data, f)



    def _load_processed_file(self, f_path: str) -> Data:
        """
        Load pre-processed file to a Data object.

        The pre-processed file is loaded into a torch-geometric Data object with the required attributes.

        :param f_path: input (absolute) file path of the preprocessed file.
        :returns: Data(x=[N, 1] (torch.float()), pos=[N, 2] (torch.float()), bbox=[L, 5] (torch.long()), file_id,
                       y=[L] (torch.long()), label=[L] (list), edge_index=[2, P] (torch.long()))
        """
        with open(f_path, 'rb') as f:
            data = pickle.load(f)

        # Ensure all required attributes are present and correctly shaped
        assert data.x.shape[1] == 1, "Node features x should have shape [N, 1]"
        assert data.pos.shape[1] == 2, "Position pos should have shape [N, 2]"
        assert len(data.bbox.shape) == 2 and data.bbox.shape[1] == 5, "Bounding boxes bbox should have shape [L, 5]"
        assert data.edge_index.shape[0] == 2, "Edge indices should have shape [2, P]"
        assert len(data.y) == data.bbox.shape[0], "Labels y should match the number of bounding boxes"

        # Optionally, add other checks or processing steps as needed
        
        return data

    def raw_files(self, mode: str) -> List[str]:
        # Fetch raw files in HDF5 format
        return glob.glob(os.path.join(self.root, mode, "*.hdf5"))

    def processed_files(self, mode: str) -> List[str]:
        # Fetch processed files in pickle format
        processed_dir = os.path.join(self.root, "processed")
        return glob.glob(os.path.join(processed_dir, mode, "*.pkl"))

    @property
    def classes(self) -> List[str]:
        return ["ball"]  # Update with your actual class names

    class HDF5Loader:
        def __init__(self, file_path):
            self.file_path = file_path
            self.event_x = list()
            self.event_y = list()
            self.event_t = list()
            self.event_p = list()

            self.gt_x = list()
            self.gt_y = list()
            self.gt_t = list()

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