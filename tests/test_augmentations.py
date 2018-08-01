from neurotorch.augmentations.brightness import Brightness
from neurotorch.augmentations.occlusion import Occlusion
from neurotorch.augmentations.duplicate import Duplicate
import unittest
from neurotorch.datasets.volumedataset import TiffVolume, AlignedVolume
import tifffile as tif
import os.path
import pytest
from neurotorch.datasets.datatypes import BoundingBox, Vector

IMAGE_PATH = "./tests/images/"


class TestAugmentations(unittest.TestCase):
    def test_brightness(self):
        input_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                "sample_volume.tif"))
        label_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                "labels.tif"))
        training_dataset = AlignedVolume((input_dataset, label_dataset),
                                         iteration_size=BoundingBox(Vector(0, 0, 0), Vector(128, 128, 20)),
                                         stride=Vector(128, 128, 20))

        brightness_dataset = Brightness(training_dataset)

        tif.imsave(os.path.join(IMAGE_PATH, "test_brightness_input.tif"),
                   brightness_dataset[10][0].getArray())
        tif.imsave(os.path.join(IMAGE_PATH, "test_brightness_label.tif"),
                   brightness_dataset[10][1].getArray()*255)

    def test_occlusion(self):
        input_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                "sample_volume.tif"))
        label_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                "labels.tif"))
        training_dataset = AlignedVolume((input_dataset, label_dataset),
                                         iteration_size=BoundingBox(Vector(0, 0, 0), Vector(128, 128, 20)),
                                         stride=Vector(128, 128, 20))

        occlusion_dataset = Occlusion(training_dataset)

        tif.imsave(os.path.join(IMAGE_PATH, "test_occlusion_input.tif"),
                   occlusion_dataset[10][0].getArray())
        tif.imsave(os.path.join(IMAGE_PATH, "test_occlusion_label.tif"),
                   occlusion_dataset[10][1].getArray()*255)

    def test_duplication(self):
        input_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                "sample_volume.tif"))
        label_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                "labels.tif"))
        training_dataset = AlignedVolume((input_dataset, label_dataset),
                                         iteration_size=BoundingBox(Vector(0, 0, 0), Vector(128, 128, 20)),
                                         stride=Vector(128, 128, 20))

        duplicate_dataset = Duplicate(training_dataset)

        tif.imsave(os.path.join(IMAGE_PATH, "test_duplicate_input.tif"),
                   duplicate_dataset[10][0].getArray())
        tif.imsave(os.path.join(IMAGE_PATH, "test_duplicate_label.tif"),
                   duplicate_dataset[10][1].getArray()*255)
