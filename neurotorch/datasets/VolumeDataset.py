from torch.utils.data import Dataset
from neurotorch.datasets.TiffDataset import TiffDataset
from functools import reduce
import numpy as np
import re


class ThreeDimDataset(Dataset):
    """
    Creates a three-dimensional volume dataset with the corresponding chunk
    size.
    """
    def __init__(self, array, chunk_size):
        self.dimensions = array.shape
        self.dtype = array.dtype
        self.chunk_size = chunk_size[::-1]  # Reverse index ordering for Numpy

        if any([chunk_size > shape for chunk_size, shape
                in zip(self.chunk_size, self.dimensions)]):
            raise ValueError("The chunk size {} must be smaller " +
                             "than the volume size {}".format(self.chunk_size,
                                                              self.dimensions))

        self.n_chunks = tuple([int(shape/chunk_size) for shape, chunk_size
                               in zip(self.dimensions, self.chunk_size)])

        self.length = reduce(lambda x, y: x*y, self.n_chunks)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        chunk_coordinate = np.unravel_index(idx, dims=self.n_chunks)
        coordinate = tuple([chunk_size*chunk for chunk_size, chunk
                            in zip(self.chunk_size, chunk_coordinate)])

        return self.array[coordinate[0]:coordinate[0]+self.chunk_size[0],
                          coordinate[1]:coordinate[1]+self.chunk_size[1],
                          coordinate[2]:coordinate[2]+self.chunk_size[2]]

    def getDimensions(self):
        return self.dimensions

    def getChunkSize(self):
        return self.chunk_size


class TwoDimDataset(ThreeDimDataset):
    """
    Creates a two-dimensional image dataset with the corresponding chunk
    size dimensions.
    """
    def __init__(self, array, chunk_size=(256, 256)):
        super().__init__(array, chunk_size=([*chunk_size, 1]))


class DatasetDelegator(Dataset):
    def __init__(self, filename):
        if re.search(r"^.*\.tif{1,2}", filename):
            self._dataset = TiffDataset(filename)
        if re.search(r"^*.h(df)5$", filename):
            self._dataset = TiffDataset(filename)
        else:
            raise ValueError("{} could not be read".format(filename))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]
