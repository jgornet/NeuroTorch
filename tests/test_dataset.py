from neurotorch.datasets.helperclasses import (DatasetSplitter,
                                               SupervisedDataset)
from neurotorch.datasets.volumedataset import TiffDataset
from neurotorch.datasets.stitcher import TiffStitcher
import unittest
import tifffile as tif
import os.path

IMAGE_PATH = "./tests/images/"


class TestDataset(unittest.TestCase):
    def test_torch_dataset(self):
        input_dataset = TiffDataset(os.path.join(IMAGE_PATH,
                                                 "sample_volume.tif"))
        label_dataset = TiffDataset(os.path.join(IMAGE_PATH,
                                                 "labels.tif"))
        dataset = SupervisedDataset(input_dataset, label_dataset)
        dataset_balancer = DatasetSplitter(dataset)
        training_dataset = dataset_balancer.getTrainDataset()

        tif.imsave(os.path.join(IMAGE_PATH, "test_input.tif"),
                   training_dataset[0]["input"])
        tif.imsave(os.path.join(IMAGE_PATH, "test_label.tif"),
                   training_dataset[0]["label"])

    def test_tiff_dataset(self):
        # Test that TiffDataset opens a TIFF stack
        testDataset = TiffDataset(os.path.join(IMAGE_PATH,
                                               "sample_volume.tif"))

        # Test that TiffDataset has the correct length
        self.assertEqual(80, len(testDataset),
                         "TIFF dataset size does not match correct size")

        # Test that TiffDataset outputs the correct samples
        self.assertTrue((tif.imread(os.path.join(IMAGE_PATH,
                                                 "test_sample.tif"))
                         == testDataset[79]).all,
                        "TIFF dataset value does not match correct value")

        # Test that TiffDataset can read and write consistent samples
        tif.imsave(os.path.join(IMAGE_PATH,
                                "test_write.tif"), testDataset[79])
        self.assertTrue((tif.imread(os.path.join(IMAGE_PATH,
                                                 "test_write.tif"))
                         == testDataset[79]).all,
                        "TIFF dataset output does not match written output")

    def test_stitcher(self):
        # Stitch a test TIFF dataset
        testDataset = TiffDataset(os.path.join(IMAGE_PATH,
                                               "sample_volume.tif"))
        stitcher = TiffStitcher(testDataset, testDataset.getDimensions())
        stitcher.stitch_dataset(testDataset,
                                os.path.join(IMAGE_PATH,
                                             "test_stitcher.tif"))
