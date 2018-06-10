from neurotorch.datasets.TiffDataset import TiffDataset
import unittest
import tifffile as tif


class TestTiffDataset(unittest.TestCase):
    def test_tiff_dataset(self):
        # Test that TiffDataset opens a TIFF stack
        testDataset = TiffDataset("./test/datasets/sample_volume.tif")

        # Test that TiffDataset has the correct length
        self.assertEqual(80, len(testDataset),
                         "TIFF dataset size does not match correct size")

        # Test that TiffDataset outputs the correct samples
        self.assertTrue((tif.imread("./test/datasets/test_sample.tif")
                         == testDataset[79]).all,
                        "TIFF dataset value does not match correct value")

        # Test that TiffDataset can read and write consistent samples
        tif.imsave("test_write.tif", testDataset[79])
        self.assertTrue((tif.imread("./test/datasets/test_write.tif")
                         == testDataset[79]).all,
                        "TIFF dataset output does not match written output")
