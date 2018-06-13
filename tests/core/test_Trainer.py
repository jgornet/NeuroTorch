import unittest
from neurotorch.core.Trainer import Trainer
from neurotorch.nets.RSUNet import RSUNet
from neurotorch.datasets.TiffDataset import TiffDataset


class TestTrainer(unittest.TestCase):
    def test_training(self):
        net = RSUNet()
        inputs_dataset = TiffDataset("./tests/datasets/sample_volume.tif")
        labels_dataset = TiffDataset("./tests/datasets/labels.tif")
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=2)
        trainer.run_training()
