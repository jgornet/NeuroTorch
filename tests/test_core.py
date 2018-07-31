import unittest
from neurotorch.core.trainer import Trainer
from neurotorch.nets.RSUNet import RSUNet
from neurotorch.datasets.volumedataset import (TorchVolume, TiffVolume)
from neurotorch.training.logging import (LossWriter,
                                         TrainingLogger)
from neurotorch.training.checkpoint import CheckpointWriter
import os.path
import os
import shutil
import pytest

IMAGE_PATH = "./tests/images"


class TestTrainer(unittest.TestCase):
    @pytest.mark.skip()
    def test_gpu_training(self):
        net = RSUNet()
        inputs_dataset = TorchVolume(TiffVolume(os.path.join(IMAGE_PATH,
                                                               "sample_volume.tif")))
        labels_dataset = TorchVolume(TiffVolume(os.path.join(IMAGE_PATH,
                                                  "labels.tif")))
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=1,
                          gpu_device=0)
        trainer.run_training()

    def test_cpu_training(self):
        net = RSUNet()
        inputs_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "sample_volume.tif"))
        labels_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "labels.tif"))
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=1)
        trainer.run_training()

    @pytest.mark.skip()
    def test_loss_writer(self):
        if not os.path.isdir('./tests/test_experiment'):
            os.mkdir('tests/test_experiment')
        shutil.rmtree('./tests/test_experiment')

        net = RSUNet()
        inputs_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "sample_volume.tif"))
        labels_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "labels.tif"))
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=1,
                          gpu_device=0)
        trainer = LossWriter(trainer, './tests/', "test_experiment")
        trainer.run_training()

    @pytest.mark.skip()
    def test_training_logger(self):
        net = RSUNet()
        inputs_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "sample_volume.tif"))
        labels_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "labels.tif"))
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=1,
                          gpu_device=0)
        trainer = TrainingLogger(trainer, logger_dir='.')
        trainer.run_training()

    @pytest.mark.skip()
    def test_checkpoint(self):
        if not os.path.isdir('./tests/checkpoints'):
            os.mkdir('tests/checkpoints')

        net = RSUNet()
        inputs_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "sample_volume.tif"))
        labels_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "labels.tif"))
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=10,
                          gpu_device=0)
        trainer = CheckpointWriter(trainer,
                                   checkpoint_dir='./tests/checkpoints',
                                   checkpoint_period=5)
        trainer.run_training()
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=10,
                          checkpoint='./tests/checkpoints/iteration_5.ckpt',
                          gpu_device=0)
        trainer.run_training()
