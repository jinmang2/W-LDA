from typing import Optional, List

import torch
from torch.utils.data import Dataset
from torch.utils.data import (
    SequentialSampler, RandomSampler, BatchSampler
)


class DocDataset(Dataset):

    def __init__(self):
        pass

    def __getiem__(self):
        pass


class BoWDataHandler:

    def __init__(
        self, 
        batch_size : int = 1,
        data_path : str = "",
        device: Union[torch.device, str] = "cpu",
    ):
        self.batch_size = batch_size
        data, labels, maps = self.load(data_path)
        self.device = device

        data_names = ['train', 'valid', 'test', 'train_with_labels',
                      'valid_with_labels', 'test_with_labels']
        label_names = ['train_label', 'valid_label', 'test_label']

        self.data = dict(zip(data_names, data))
        self.labels = dict(zip(label_names, labels))

        # repeat data to at least match batch_size
        for k, v in self.data.items():
            if v is not None and v.shape[0] < self.batch_size:
                print('NOTE: Number of samples for {0} is smaller than batch_size ({1}<{2}). Duplicating samples to exceed batch_size.'.format(
                    k, v.shape[0], self.batch_size))
                if type(v) is np.ndarray:
                    self.data[k] = np.tile(
                        v, (self.batch_size // v.shape[0] + 1, 1))
