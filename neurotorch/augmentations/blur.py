from neurotorch.augmentations.augmentation import Augmentation
from neurotorch.datasets.dataset import Data
from scipy.ndimage.filters import gaussian_filter
from numpy.random import normal
from scipy.ndimage.filters import median_filter
from random import random


class Blur(Augmentation):
    def __init__(self, volume, max_blur=(0.66, 4, 4), **kwargs):
        self.setMaxBlur(max_blur)
        super().__init__(volume, **kwargs)

    def augment(self, bounding_box):
        raw = self.getInput(bounding_box)
        label = self.getLabel(bounding_box)
        augmented_raw, augmented_label = self.blur(raw, label, self.max_blur)

        return (augmented_raw, augmented_label)

    def setFrequency(self, frequency):
        self.frequency = frequency

    def setMaxBlur(self, max_blur):
        self.max_blur = max_blur

    def blur(self, raw_data, label_data, max_blur):
        raw = raw_data.getArray().copy()

        gaussian_raw = gaussian_filter(raw, sigma=max_blur)
        noise = raw - median_filter(raw, size=(3, 3, 3))

        gaussian_raw = gaussian_raw + noise
        gaussian_raw = gaussian_raw.astype(raw.dtype)

        augmented_raw_data = Data(guassian_raw, raw_data.getBoundingBox())

        return augmented_raw_data, label_data
