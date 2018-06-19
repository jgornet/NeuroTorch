import numpy as np
import tifffile as tif
from neurotorch.datasets.volumedataset import ThreeDimDataset


class DatasetStitcher:
    def stitch_dataset(self, dataset: ThreeDimDataset):
        """
        Stitches a VolumeDataset into a numpy array

        :param dataset: A VolumeDataset to stitch
        """
        array = np.zeros(dataset.dimensions, dtype=dataset.dtype)
        z_size, y_size, x_size = dataset.getChunkSize()

        for index, sample in enumerate(dataset):
            z_chunk, y_chunk, x_chunk = np.unravel_index(index,
                                                         dims=dataset.n_chunks)
            x = x_size*x_chunk
            y = y_size*y_chunk
            z = z_size*z_chunk

            array[z:z+z_size, y:y+y_size, x:x+x_size] = sample

        return array


class TiffStitcher(DatasetStitcher):
    def stitch_dataset(self, dataset, tiff_file):
        """
        Stitches a VolumeDataset and saves it as a TIFF stack

        :param dataset: A VolumeDataset to stitch
        :param tiff_file: The file path of the TIFF reconstruction
        """
        array = super().stitch_dataset(dataset)
        tif.imsave(tiff_file, array)
