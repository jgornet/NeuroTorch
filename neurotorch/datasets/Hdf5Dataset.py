import h5py
from neurotorch.datasets.VolumeDataset import ThreeDimDataset


class Hdf5Dataset(ThreeDimDataset):
    """
    Creates a dataset from a HDF5 file
    """
    def __init__(self, hdf5_file, dataset, chunk_size=(256, 256, 20)):
        self.hdf5_file = h5py.File(hdf5_file)
        self.array = self.hdf5_file[dataset].value

        super(self.array, chunk_size)
