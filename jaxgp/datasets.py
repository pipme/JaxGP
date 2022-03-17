import numpy as np
from torch.utils import data
from torch.utils.data import Dataset as DatasetTorch

from .helpers import Array, dataclass, field


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


@dataclass
class Dataset:
    X: Array
    Y: Array = None

    def __post_init__(self):
        if self.Y.ndim == 1:
            self.Y = self.Y[..., None]
        if self.X.ndim == 1:
            self.X = self.X[..., None]
        assert self.X.ndim == 2
        assert self.Y.ndim == 2

    def __repr__(self) -> str:
        return (
            f"- Number of datapoints: {self.X.shape[0]}\n- Dimension:"
            f" {self.X.shape[1]}"
        )

    @property
    def N(self) -> int:
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        return self.X.shape[1]

    @property
    def out_dim(self) -> int:
        return self.Y.shape[1]


class CustomDataset(DatasetTorch):
    def __init__(self, X, Y, transform=None, target_transform=None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :]


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


if __name__ == "__main__":
    d = Dataset(X=None)
    print(d)
