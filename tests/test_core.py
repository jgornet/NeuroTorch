import unittest
from neurotorch.core.Trainer import Trainer
from neurotorch.nets.RSUNet import RSUNet
from neurotorch.datasets.TiffDataset import TiffDataset
import os.path

IMAGE_PATH = "./tests/images"


class TestTrainer(unittest.TestCase):
    def test_training(self):
        net = RSUNet()
        inputs_dataset = TiffDataset(os.path.join(IMAGE_PATH,
                                                  "sample_volume.tif"))
        labels_dataset = TiffDataset(os.path.join(IMAGE_PATH,
                                                  "labels.tif"))
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=2)
        trainer.run_training()
