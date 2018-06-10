from neurotorch.datasets.DatasetBalancer import (DatasetBalancer,
                                                 PairedDataset)
from neurotorch.datasets.TiffDataset import TiffDataset
import unittest
import tifffile as tif


class TestTrainingDataset(unittest.TestCase):
    def test_DatasetLoader(self):
        input_dataset = TiffDataset("./test/datasets/sample_volume.tif")
        label_dataset = TiffDataset("./test/datasets/labels.tif")

        dataset = PairedDataset(input_dataset, label_dataset)
        dataset_balancer = DatasetBalancer(dataset)
        training_dataset = dataset_balancer.getTrainDataset()

        tif.imsave("./test/datasets/test_input.tif",
                   training_dataset[0]["input"])
        tif.imsave("./test/datasets/test_label.tif",
                   training_dataset[0]["label"])
