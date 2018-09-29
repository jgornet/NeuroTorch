from neurotorch.datasets.dataset import (AlignedVolume, Array)
from neurotorch.datasets.filetypes import (LargeTiffVolume, TiffVolume)
import numpy as np
import unittest
import tifffile as tif
import os.path
import pytest
from os import getpid
from psutil import Process
from neurotorch.datasets.datatypes import BoundingBox, Vector
import time

IMAGE_PATH = "./tests/images/"


class TestDataset(unittest.TestCase):
    def test_torch_dataset(self):
        input_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                "sample_volume.tif"))
        label_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                "labels.tif"))
        training_dataset = AlignedVolume((input_dataset, label_dataset),
                                         iteration_size=BoundingBox(Vector(0, 0, 0), Vector(128, 128, 20)),
                                         stride=Vector(128, 128, 20))

        tif.imsave(os.path.join(IMAGE_PATH, "test_input.tif"),
                   training_dataset[10][0].getArray())
        tif.imsave(os.path.join(IMAGE_PATH, "test_label.tif"),
                   training_dataset[10][1].getArray()*255)

    def test_tiff_dataset(self):
        # Test that TiffVolume opens a TIFF stack
        testDataset = TiffVolume(os.path.join(IMAGE_PATH,
                                              "sample_volume.tif"),
                                 iteration_size=BoundingBox(Vector(0, 0, 0), Vector(128, 128, 20)),
                                 stride=Vector(128, 128, 20))

        # Test that TiffVolume has the correct length
        self.assertEqual(64, len(testDataset),
                         "TIFF dataset size does not match correct size")

        # Test that TiffVolume outputs the correct samples
        self.assertTrue((tif.imread(os.path.join(IMAGE_PATH,
                                                 "test_sample.tif"))
                         == testDataset[10].getArray()).all,
                        "TIFF dataset value does not match correct value")

        # Test that TiffVolume can read and write consistent samples
        tif.imsave(os.path.join(IMAGE_PATH,
                                "test_write.tif"), testDataset[10].getArray())
        self.assertTrue((tif.imread(os.path.join(IMAGE_PATH,
                                                 "test_write.tif"))
                         == testDataset[10].getArray()).all,
                        "TIFF dataset output does not match written output")

    def test_stitcher(self):
        # Stitch a test TIFF dataset
        inputDataset = TiffVolume(os.path.join(IMAGE_PATH,
                                               "sample_volume.tif"))
        outputDataset = Array(np.zeros(inputDataset
                                        .getBoundingBox()
                                        .getNumpyDim()))
        for data in inputDataset:
            outputDataset.blend(data)

        self.assertTrue((inputDataset[20].getArray()
                         == outputDataset[20].getArray()).all,
                        "Blending output does not match input")

        tif.imsave(os.path.join(IMAGE_PATH,
                                "test_stitch.tif"),
                   outputDataset[100]
                   .getArray()
                   .astype(np.uint16))

    def test_large_dataset(self):
        # Test that TiffVolume opens a TIFF stack
        testDataset = LargeTiffVolume(os.path.join(IMAGE_PATH,
                                                   "test_large_volume"),
                                      iteration_size=BoundingBox(Vector(0, 0, 0),
                                                                 Vector(128, 128, 20)),
                                      stride=Vector(128, 128, 20))

        # Test that TiffVolume can read and write consistent samples
        tif.imsave(os.path.join(IMAGE_PATH,
                                "test_large_write.tif"),
                   testDataset[0].getArray())

    def test_memory_free(self):
        # Test that
        process = Process(getpid())
        initial_memory = process.memory_info().rss
        
        start = time.perf_counter()
        with TiffVolume(os.path.join(IMAGE_PATH, "sample_volume.tif"),
                        BoundingBox(Vector(0, 0, 0),
                                    Vector(1024, 512, 50))) as v:
            volume_memory = process.memory_info().rss
        end = time.perf_counter()
        print("Load time: {} secs".format(end-start))

        final_memory = process.memory_info().rss

        self.assertAlmostEqual(initial_memory, final_memory,
                               delta=initial_memory*0.2,
                               msg=("memory leakage: final memory usage is " +
                                    "larger than the initial memory usage"))
        self.assertLess(initial_memory, volume_memory,
                        msg=("volume loading error: volume memory usage is " +
                             "not less than the initial memory usage"))
