from neurotorch.datasets.DatasetStitcher import TiffStitcher
from neurotorch.datasets.TiffDataset import TiffDataset
import unittest


class test_TiffStitcher(unittest.TestCase):
    def test_stitcher(self):
        # Stitch a test TIFF dataset
        testDataset = TiffDataset("./test/datasets/sample_volume.tif")
        stitcher = TiffStitcher()
        stitcher.stitch_dataset(testDataset,
                                "./test/datasets/test_stitcher.tif")
