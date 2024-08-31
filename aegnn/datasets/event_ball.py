from typing import Callable, List, Tuple
from .base.event_dm import EventDataModule
import glob
import os
import h5py
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from .utils.normalization import normalize_time
from torch_geometric.nn.pool import radius_graph
import pickle


class EventBall(EventDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        transform: Callable[[Any], Any] | None = None,
    ):
        super().__init__(
            img_shape=(1280, 720),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            transform=transform,
        )

        self.radius = 3.0
        self.max_neighbors = 128
        self.n_samples = 25.000
        self.sampling = True

    def _prepare_dataset(self, mode: str):
        raw_files = self.raw_files(mode)
        print(f"Found {len(raw_files)} files in dataset (mode = {mode})")
        total_bbox_count = self._total_bbox_count(raw_files)
        print(f"Total count of (filtered) bounding boxes = {total_bbox_count}")

        for rf in tqdm(raw_files):
            self._processing(rf, self.root, mode)

    def raw_files(self, mode: str) -> List[str]:
        glob.glob(os.path.join(self.root, mode, "*.hdf5"))

    def _total_bbox_count(self, raw_files: List[str]) -> int:
        """
        :param raw_files: List of event data files

        :return num_bbox: Count of bounding boxes in all files specified in `raw_files`
        """
        num_bbox = 0
        for rf in raw_files:
            data_loader = HDF5Loader(rf)
            bounding_boxes = data_loader.get_bbox()
            num_bbox += len(bounding_boxes)

        return num_bbox

    def _processing(self, rf: str, root: str):
        data_loader = HDF5Loader(rf)

        bounding_boxes = data_loader.get_bbox()
        labels = np.array(["ball" for i in range(len(bounding_boxes[0]))])

        for i, bbox in enumerate(bounding_boxes):
            processed_dir = os.path.join(root, "processed")
            processed_file = rf.replace(root, processed_dir).replace(
                ".hdf5", f"{i}.pkl"
            )
            if os.path.exists(processed_file):
                continue

            # Determine temporal window around current bouding box [t_start, t_end] and all of the bounding boxes within this window
            sample_dict = dict()
            t_bbox = bbox[0]
            t_start = t_bbox - 100.000
            t_end = t_bbox + 300.000
            bbox_mask = np.logical_and(
                t_start < bounding_boxes["t"], bounding_boxes["t"] < t_end
            )

            sample_dict["bbox"] = torch.tensor(bounding_boxes[bbox_mask].tolist())
            sample_dict["label"] = labels[bbox_mask]
            sample_dict["raw_file"] = rf

            # Load raw data around bounding box
            idx_start, end_idx, data = data_loader.load_chunk(t_start, t_end - t_start)
            sample_dict["raw"] = (idx_start, end_idx)

            # Continue if the number of events is too small (end or start of file)
            if data.size < 4.000:
                continue

            # normalize data to same number of events per sample
            num_samples = self.n_samples
            if data.size <= num_samples:
                sample_idx = np.arange(data.size)
            elif self.sampling:
                sample_idx = np.random.choice(
                    np.arange(data.size), size=num_samples, replace=False
                )
            else:
                sample_idx = np.arange(num_samples)

            sample = self.buffer_to_data(data[sample_idx])
            sample_dict["sample_idx"] = sample_idx

            # Graph generation
            device = torch.device(torch.cuda.current_device())
            sample.pos[:, 2] = normalize_time(sample.pos[:, 2])
            edge_index = radius_graph(
                sample.pos.to(device),
                r=self.radius,
                max_num_neighbors=self.max_neighbors,
            )
            sample_dict["edge_index"] = edge_index.cpu()

            # store resulting dictionary in file, with data to recreate graph
            os.makedirs(os.path.dirname(processed_file), exist_ok=True)
            with open(processed_file, "wb") as f:
                pickle.dump(sample_dict, f)

    def _load_processed_file(self, f_path: str) -> Data:
        with open(f_path, "rb") as f:
            data_dict = pickle.load(f)

        data_loader = HDF5Loader(data_dict["raw_file"])
        raw_start, raw_end = data_dict["raw"]
        data = data_loader.load_n_events(raw_start, raw_end)
        data = data[data_dict["sample_idx"]]

        data = self.buffer_to_data(data, label=data_dict["label"], file_id=f_path)
        data.bbox = data_dict["bbox"][:, 1:6].long()  # (x, y, w, h, class_id)
        data.y = data.bbox[:, -1]
        data.pos[:, 2] = normalize_time(data.pos[:, 2])
        data.edge_index = data_dict["edge_index"]
        return data

    def buffer_to_data(
        self, buffer: List[Tuple[float, int, int, int]], **kwargs
    ) -> Data:
        x = torch.from_numpy(buffer["x"].astype(np.float32))
        y = torch.from_numpy(buffer["y"].astype(np.float32))
        t = torch.from_numpy(buffer["t"].astype(np.float32))
        p = torch.from_numpy(buffer["p"].astype(np.float32)).view(-1, 1)
        pos = torch.stack([x, y, t], dim=1)
        return Data(x=p, pos=pos, **kwargs)


class HDF5Loader:
    def __init__(self, file_path: str):
        self.hdf_file = self.HDF5File(file_path)
        event_t, _, _, _ = self.hdf_file.get_events()
        self._ev_count = len(event_t)

    def total_time(self) -> float:
        """
        :return t: The last time stamp in the even data file
        """
        t, _, _, _ = self.hdf_file.get_events()
        return t[-1]

    def load_n_events(
        self, start_idx: int, end_idx: int
    ) -> Tuple[int, List[Tuple[float, int, int, int]]]:
        """
        Returns all events between two indices

        :param start_idx: First event
        :param end_idx: Last event
        :return: Buffer of event data between start_idx and end_idx
        """
        return self.get_events()[start_idx:end_idx]

    def load_chunk(
        self, start_time: int, time_offset: float
    ) -> Tuple[int, int, List[Tuple[float, int, int, int]]]:
        """
        Returns the chunk of event data, starting from start_time to the wished time_offset

        :param start_time: The beginning time value
        :param time_offset: The offset of the beginning time value
        :return: Start index of buffer data, end index of buffer data and buffered data
        """

        start_idx = self.seek_time(start_time)
        start_time = self.hdf_file.get_events()[0][start_idx]
        end_index = self.seek_time(start_time + time_offset)

        return start_idx, end_index, self.get_events()[start_idx:end_index]

    def seek_time(self, start_time: float, term_criterion: int = 100.000) -> int:
        """
        Find the index of the closest time in the event data for start_time

        :param start_time: The desired start time of the event data
        :return index: The index of the closest event time
        """
        if start_time > self.total_time():
            return self._ev_count - 1
        if start_time <= 0:
            return 0

        event_t, _, _, _ = self.hdf_file.get_events()

        low = 0
        high = self._ev_count

        while high - low > term_criterion:
            middle = (low + high) // 2

            mid = event_t[middle]

            if mid > start_time:
                high = middle
            elif mid < start_time:
                low = middle + 1
            else:
                return middle

        # we now know it is between low and high
        final_buffer = np.array(event_t[low:high])
        final_index = np.searchsorted(final_buffer, start_time)
        return low + final_index

    def get_bbox(self) -> List[Tuple[float, int, int, int, int, int]]:
        """
        Return all bounding boxes in the file with a set widht and height and the same class id.

        :return bboxes: (t, x, y, w, h, class_id)
        """
        gt_t, gt_x, gt_y = self.hdf_file.get_gt()

        bboxes = np.empty(
            len(gt_t),
            dtype=[
                ("t", "f8"),
                ("x", "u2"),
                ("y", "u2"),
                ("w", "u2"),
                ("h", "u2"),
                ("class_id", "u2"),
            ],
        )

        bboxes["t"] = np.array(gt_t, dtype=np.float64)
        bboxes["x"] = np.array(gt_x, dtype=np.uint16)
        bboxes["y"] = np.array(gt_y, dtype=np.uint16)
        bboxes["w"] = np.full(len(gt_t), 10, dtype=np.uint16)
        bboxes["h"] = np.full(len(gt_t), 10, dtype=np.uint16)
        bboxes["class_id"] = np.full(len(gt_t), 0, dtype=np.uint16)

        return bboxes

    def get_events(self) -> List[Tuple[float, int, int, int]]:
        """
        Return all events in the file

        :return bboxes: (t, x, y, p)
        """
        event_t, event_x, event_y, event_p = self.hdf_file.get_events()

        events = np.empty(
            len(event_t),
            dtype=[
                ("t", "f8"),
                ("x", "u2"),
                ("y", "u2"),
                ("p", "u2"),
            ],
        )

        events["t"] = np.array(event_t, dtype=np.float64)
        events["x"] = np.array(event_x, dtype=np.uint16)
        events["y"] = np.array(event_y, dtype=np.uint16)
        events["p"] = np.array(event_p, dtype=np.uint16)

        return events

    class HDF5File:
        def __init__(self, file_path: str):
            self.file_path = file_path
            with h5py.File(self.file_path, "r") as hdf5:
                self.event_x = hdf5["event_x"][:]
                self.event_y = hdf5["event_y"][:]
                self.event_t = hdf5["event_t"][:]
                self.event_p = hdf5["event_p"][:]

                self.gt_x = hdf5["gt_x"][:]
                self.gt_y = hdf5["gt_y"][:]
                self.gt_t = hdf5["gt_t"][:]

                # Translate from unix epoch time to real microseconds in dataset
                min_event_t = self.event_t[0]
                self.event_t = self.event_t - min_event_t
                self.gt_t = self.gt_t - min_event_t

        def get_events(self) -> Tuple[List[float], List[int], List[int], List[int]]:
            """
            :return: (t, x, y, p)
            """
            return (self.event_t, self.event_x, self.event_y, self.event_p)

        def get_gt(self) -> Tuple[List[float], List[int], List[int]]:
            """
            :return: (t, x, y)
            """
            return (self.gt_t, self.gt_x, self.gt_y)
