import numpy as np
import tifffile as tif
from neurotorch.datasets.volumedataset import VolumeDataset


class DatasetStitcher:
    def __init__(self, dataset, dimensions, chunk_size=None, dtype=None,
                 n_chunk_dim=None):
        self.dataset = dataset
        self.dimensions = dimensions

        if chunk_size is None:
            self.chunk_size = dataset.chunk_size
        else:
            self.chunk_size = chunk_size

        if n_chunk_dim is None:
            self.n_chunk_dim = dataset.n_chunks
        else:
            self.n_chunk_dim = n_chunk_dim

        array_size = tuple([round(dim/chunk)*chunk_size
                            for dim, chunk in zip(dimensions, chunk_size)])

        if dtype is None:
            self.array = np.zeros(array_size, dtype=dataset.dtype)
        else:
            self.array = np.zeros(array_size, dtype=dtype)

    def stitch_dataset(self, dataset: VolumeDataset):
        """
        Stitches a VolumeDataset into a numpy array

        :param dataset: A VolumeDataset to stitch
        """
        for index, sample in enumerate(dataset):
            self.add_sample(sample, index)

        dim = self.dimensions
        return self.array[0:dim[0], 0:dim[1], 0:dim[2]]

    def add_sample(self, sample, index):
        z_size, y_size, x_size = self.chunk_size
        z_chunk, y_chunk, x_chunk = np.unravel_index(index,
                                                     dims=self.n_chunk_dim)

        x = x_size*x_chunk
        y = y_size*y_chunk
        z = z_size*z_chunk

        self.array[z:z+z_size, y:y+y_size, x:x+x_size] = sample


class TiffStitcher(DatasetStitcher):
    def stitch_dataset(self, dataset, tiff_file):
        """
        Stitches a VolumeDataset and saves it as a TIFF stack

        :param dataset: A VolumeDataset to stitch
        :param tiff_file: The file path of the TIFF reconstruction
        """
        super().stitch_dataset(dataset)
        tif.imsave(tiff_file, self.array)
