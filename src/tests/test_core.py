import unittest
from neurotorch.loss.SimplePointWeighting import SimplePointBCEWithLogitsLoss
from neurotorch.core.trainer import Trainer
from neurotorch.nets.RSUNet import RSUNet
from neurotorch.datasets.datatypes import (BoundingBox, Vector)
from neurotorch.datasets.dataset import Array
from neurotorch.datasets.filetypes import TiffVolume
from neurotorch.training.logging import (LossWriter,
                                         TrainingLogger)
from neurotorch.training.checkpoint import CheckpointWriter
import os.path
import os
import shutil
import pytest
import tifffile as tif
import numpy as np
from neurotorch.core.predictor import Predictor
import time

IMAGE_PATH = "./tests/images"


class TestTrainer(unittest.TestCase):
    def test_gpu_training(self):
        net = RSUNet()
        inputs_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "inputs.tif"),
                                    BoundingBox(Vector(0, 0, 0),
                                                Vector(1024, 512, 50)))
        labels_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "labels.tif"),
                                    BoundingBox(Vector(0, 0, 0),
                                                Vector(1024, 512, 50)))
        inputs_dataset.__enter__()
        labels_dataset.__enter__()
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=1,
                          gpu_device=1)
        trainer.run_training()

    @pytest.mark.skip()
    def test_cpu_training(self):
        net = RSUNet()
        inputs_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "inputs.tif"),
                                    BoundingBox(Vector(0, 0, 0),
                                                Vector(1024, 512, 50)))
        labels_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "labels.tif"),
                                    BoundingBox(Vector(0, 0, 0),
                                                Vector(1024, 512, 50)))
        inputs_dataset.__enter__()
        labels_dataset.__enter__()
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=1)
        trainer.run_training()

    def test_loss_writer(self):
        if not os.path.isdir('./tests/test_experiment'):
            os.mkdir('tests/test_experiment')
        shutil.rmtree('./tests/test_experiment')

        net = RSUNet()
        inputs_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "inputs.tif"),
                                    BoundingBox(Vector(0, 0, 0),
                                                Vector(1024, 512, 50)))
        labels_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "labels.tif"),
                                    BoundingBox(Vector(0, 0, 0),
                                                Vector(1024, 512, 50)))
        inputs_dataset.__enter__()
        labels_dataset.__enter__()
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=1,
                          gpu_device=1)
        trainer = LossWriter(trainer, './tests/', "test_experiment")
        trainer.run_training()

    def test_training_logger(self):
        net = RSUNet()
        inputs_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "inputs.tif"),
                                    BoundingBox(Vector(0, 0, 0),
                                                Vector(1024, 512, 50)))
        labels_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "labels.tif"),
                                    BoundingBox(Vector(0, 0, 0),
                                                Vector(1024, 512, 50)))
        inputs_dataset.__enter__()
        labels_dataset.__enter__()
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=1,
                          gpu_device=1)
        trainer = TrainingLogger(trainer, logger_dir='.')
        trainer.run_training()

    def test_checkpoint(self):
        if not os.path.isdir('./tests/checkpoints'):
            os.mkdir('tests/checkpoints')

        net = RSUNet()
        inputs_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "inputs.tif"),
                                    BoundingBox(Vector(0, 0, 0),
                                                Vector(1024, 512, 50)))
        labels_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "labels.tif"),
                                    BoundingBox(Vector(0, 0, 0),
                                                Vector(1024, 512, 50)))
        inputs_dataset.__enter__()
        labels_dataset.__enter__()
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=10,
                          gpu_device=1)
        trainer = CheckpointWriter(trainer,
                                   checkpoint_dir='./tests/checkpoints',
                                   checkpoint_period=5)
        trainer.run_training()
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=10,
                          checkpoint='./tests/checkpoints/iteration_5.ckpt',
                          gpu_device=1)
        trainer.run_training()

    def test_loss(self):
        net = RSUNet()
        inputs_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "inputs.tif"),
                                    BoundingBox(Vector(0, 0, 0),
                                                Vector(1024, 512, 50)))
        labels_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "labels.tif"),
                                    BoundingBox(Vector(0, 0, 0),
                                                Vector(1024, 512, 50)))
        inputs_dataset.__enter__()
        labels_dataset.__enter__()
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=10,
                          gpu_device=1,
                          criterion=SimplePointBCEWithLogitsLoss())
        trainer.run_training()

    def test_prediction(self):
        if not os.path.isdir('./tests/checkpoints'):
            os.mkdir('tests/checkpoints')

        net = RSUNet()

        checkpoint = './tests/checkpoints/iteration_10.ckpt'
        inputs_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                 "inputs.tif"),
                                    BoundingBox(Vector(0, 0, 0),
                                                Vector(1024, 512, 50)))
        inputs_dataset.__enter__()
        predictor = Predictor(net, checkpoint, gpu_device=1)

        output_volume = Array(np.zeros(inputs_dataset
                                        .getBoundingBox()
                                        .getNumpyDim()))

        predictor.run(inputs_dataset, output_volume, batch_size=5)

        tif.imsave(os.path.join(IMAGE_PATH,
                                "test_prediction.tif"),
                   output_volume.getArray().astype(np.float32))
