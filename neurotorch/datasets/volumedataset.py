from torch.utils.data import Dataset as _TorchDataset
from functools import reduce
import numpy as np
import os.path
import os
import fnmatch
import tifffile as tif
import h5py
from abc import (ABC, abstractmethod)


class Dataset(ABC):
    def __init__(self):
        super.__init__()

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class TorchDataset(_TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        super.__init__()

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)


class VolumeDataset(Dataset):
    """
    Creates a three-dimensional volume dataset with the corresponding chunk
    size.
    """
    def __init__(self, array, chunk_size):
        """
        Initializes the dataset with a Numpy array and chunk size

        :param array: A three-dimensional Numpy array
        :param chunk_size: The subvolume size of each sample
        """
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

        result = self.array[coordinate[0]:coordinate[0]+self.chunk_size[0],
                            coordinate[1]:coordinate[1]+self.chunk_size[1],
                            coordinate[2]:coordinate[2]+self.chunk_size[2]]
        result = result.reshape(1, *self.chunk_size)

        return result

    def getDimensions(self):
        """
        Returns the dimensions of the dataset

        :return: The dimensions of the dataset
        """
        return self.dimensions

    def getChunkSize(self):
        """
        Returns the dimensions of the sample subvolume

        :return: The dimensions of the sample subvolume
        """
        return self.chunk_size


class TiffDataset(VolumeDataset):
    """
    Creates a dataset from a TIFF file
    """
    def __init__(self, tiff_file, chunk_size=(256, 256, 20)):
        """
        Loads a TIFF stack file or a directory of TIFF files and creates a
corresponding three-dimensional volume dataset
        :param tiff_file: Either a TIFF stack file or a directory containing TIFF files
        :param chunk_size: Dimensions of the sample subvolume
        """
        if os.path.isfile(tiff_file):
            try:
                self.array = tif.imread(tiff_file)

            except IOError:
                raise IOError("TIFF file {} could not be " +
                              "opened".format(tiff_file))

        elif os.path.isdir(tiff_file):
            tiff_list = os.listdir(tiff_file)
            tiff_list = filter(lambda f: fnmatch.fnmatch(f, '*.tif'),
                               tiff_list)

            if tiff_list:
                self.array = tif.TiffSequence(tiff_list).asarray()

        else:
            raise IOError("{} was not found".format(tiff_file))

        super().__init__(self.array, chunk_size)


class Hdf5Dataset(VolumeDataset):
    """
    Creates a dataset from a HDF5 file
    """
    def __init__(self, hdf5_file, dataset, chunk_size=(256, 256, 20)):
        """
        Loads a HDF5 dataset and creates a corresponding three-dimensional volume dataset
        :param hdf5_file: A HDF5 file path
        :param dataset: A HDF5 dataset name
        :param chunk_size: Dimensions of the sample subvolume
        """
        self.hdf5_file = h5py.File(hdf5_file)
        self.array = self.hdf5_file[dataset].value

        super(self.array, chunk_size)
