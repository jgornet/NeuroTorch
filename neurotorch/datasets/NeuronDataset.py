from torch.utils.data import Dataset
from functools import reduce
import numpy as np


class ThreeDimDataset(Dataset):
    def __init__(self, array, chunk_size):
        self.shape = array.shape
        self.chunk_size = chunk_size[::-1]

        if any([chunk_size > shape for chunk_size, shape
                in zip(self.chunk_size, self.shape)]):
            raise ValueError("The chunk size {} must be smaller " +
                             "than the volume size {}".format(self.chunk_size,
                                                              self.shape))

        self.n_chunks = tuple([int(shape/chunk_size) for shape, chunk_size
                               in zip(self.shape, self.chunk_size)])

        self.length = reduce(lambda x, y: x*y, self.n_chunks)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        chunk_coordinate = np.unravel_index(idx, dims=self.n_chunks)
        coordinate = tuple([chunk_size*chunk for chunk_size, chunk
                            in zip(self.chunk_size, chunk_coordinate)])

        return self.array[coordinate[0]:coordinate[0]+self.chunk_size[0],
                          coordinate[1]:coordinate[1]+self.chunk_size[1],
                          coordinate[2]:coordinate[2]+self.chunk_size[2]]


class TwoDimDataset(ThreeDimDataset):
    def __init__(self, array, chunk_size=(256, 256)):
        super().__init__(array, chunk_size=([*chunk_size, 1]))
