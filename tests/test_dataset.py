from neurotorch.datasets.dataset import (AlignedVolume, Array, PooledVolume)
from neurotorch.datasets.filetypes import (TiffVolume, Hdf5Volume)
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
                                                "sample_volume.tif"),
                                   BoundingBox(Vector(0, 0, 0),
                                               Vector(1024, 512, 50)))
        label_dataset = TiffVolume(os.path.join(IMAGE_PATH,
                                                "labels.tif"),
                                   BoundingBox(Vector(0, 0, 0),
                                               Vector(1024, 512, 50)))
        input_dataset.__enter__()
        label_dataset.__enter__()
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
                                 BoundingBox(Vector(0, 0, 0),
                                             Vector(1024, 512, 50)),
                                 iteration_size=BoundingBox(Vector(0, 0, 0), Vector(128, 128, 20)),
                                 stride=Vector(128, 128, 20))
        testDataset.__enter__()

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
                                               "sample_volume.tif"),
                                  BoundingBox(Vector(0, 0, 0),
                                              Vector(1024, 512, 50)))
        outputDataset = Array(np.zeros(inputDataset
                                        .getBoundingBox()
                                        .getNumpyDim()))
        inputDataset.__enter__()
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

    def test_memory_free(self):
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

    def test_pooled_volume(self):
        pooled_volume = PooledVolume(stack_size=5)
        pooled_volume.add(TiffVolume(os.path.join(IMAGE_PATH,
                                                  "sample_volume.tif"),
                                     BoundingBox(Vector(0, 0, 0),
                                                 Vector(1024, 512, 50))))
        pooled_volume.add(TiffVolume(os.path.join(IMAGE_PATH,
                                                  "sample_volume.tif"),
                                     BoundingBox(Vector(0, 0, 50),
                                                 Vector(1024, 512, 100))))
        output = pooled_volume.get(BoundingBox(Vector(0, 0, 40),
                                               Vector(128, 128, 60)))

        self.assertTrue((tif.imread(os.path.join(IMAGE_PATH,
                                                 "test_pooled_volume.tif"))
                         == output.getArray()).all,
                        "PooledVolume output does not match test case")

    def test_hdf5_volume(self):
        pooled_volume = PooledVolume(stack_size=5)
        pooled_volume.add(Hdf5Volume(os.path.join(IMAGE_PATH,
                                                  "sample_volume.h5"),
                                     "input-1",
                                     BoundingBox(Vector(0, 0, 0),
                                                 Vector(1024, 512, 50))))
        pooled_volume.add(Hdf5Volume(os.path.join(IMAGE_PATH,
                                                  "sample_volume.h5"),
                                     "input-2",
                                     BoundingBox(Vector(0, 0, 50),
                                                 Vector(1024, 512, 100))))
        output = pooled_volume.get(BoundingBox(Vector(0, 0, 40),
                                               Vector(128, 128, 60)))

        self.assertTrue((tif.imread(os.path.join(IMAGE_PATH,
                                                 "test_pooled_volume.tif"))
                         == output.getArray()).all,
                        "PooledVolume output does not match test case")
