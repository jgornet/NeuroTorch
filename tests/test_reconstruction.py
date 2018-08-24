import unittest
from neurotorch.reconstruction.app2.app2 import reconstruct
from os.path import (abspath, join)
from hashlib import md5

IMAGE_PATH = "./tests/images"
MORPH_PATH = "./tests/morphology"


class TestReconstruction(unittest.TestCase):
    def test_reconstruction(self):
        segmentation_path = abspath(join(IMAGE_PATH, "segmentation.tif"))
        reconstruct(segmentation_path)

        with open(join(MORPH_PATH, "segmentation.swc"), 'rb') as f:
            correct_reconstruction = md5(f.read()).hexdigest()
            print(correct_reconstruction)

        with open("./segmentation_hp.swc", 'rb') as f:
            test_reconstruction = md5(f.read()).hexdigest()
            print(test_reconstruction)

        self.assertEqual(correct_reconstruction, test_reconstruction)
