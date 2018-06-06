from NeuronDataset import ThreeDimDataset
import os.path
import os
import fnmatch
import tifffile as tif
import unittest
import numpy as np


class TiffDataset(ThreeDimDataset):
    """
    Creates a dataset from a TIFF file
    """
    def __init__(self, tiff_file, chunk_size=(256, 256, 20)):
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


def test_tiff_dataset():
    test = TiffDataset("Z21-2000-2200_labels.tif")
    print(len(test))
    tif.imsave("test.tif", 255*test[40].astype(np.uint8))
    print(test[40])
