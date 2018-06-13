from neurotorch.datasets.DatasetBalancer import (DatasetSplitter,
                                                 SupervisedDataset)
from neurotorch.datasets.TiffDataset import TiffDataset
import unittest
import tifffile as tif


class TestTrainingDataset(unittest.TestCase):
    def test_DatasetLoader(self):
        input_dataset = TiffDataset("./tests/datasets/sample_volume.tif")
        label_dataset = TiffDataset("./tests/datasets/labels.tif")

        dataset = SupervisedDataset(input_dataset, label_dataset)
        dataset_balancer = DatasetSplitter(dataset)
        training_dataset = dataset_balancer.getTrainDataset()

        tif.imsave("./tests/datasets/test_input.tif",
                   training_dataset[0]["input"])
        tif.imsave("./tests/datasets/test_label.tif",
                   training_dataset[0]["label"])
