from neurotorch.augmentations.augmentation import Augmentation
from neurotorch.datasets.volumedataset import Data
import random
import numpy as np


class Duplicate(Augmentation):
    def __init__(self, volume, frequency=0.1, max_slices=2):
        self.setFrequency(frequency)
        self.setMaxSlices(max_slices)
        super().__init__(volume)

    def augment(self, bounding_box):
        slices = self.getSlices()
        end = bounding_box.getSize().getComponents()[0]
        location = random.randrange(end-slices)

        raw_data = self.getInput(bounding_box)
        label_data = self.getLabel(bounding_box)
        augmented_raw, augmented_label = self.duplication(raw_data, label_data,
                                                          location=location,
                                                          slices=slices)

        return (augmented_raw, augmented_label)

    def setFrequency(self, frequency):
        self.frequency = frequency

    def setMaxSlices(self, max_slices):
        self.max_slices = max_slices

    def getMaxSlices(self):
        return self.max_slices

    def getSlices(self):
        return random.randrange(self.getMaxSlices())

    def duplication(self, raw_data, label_data,
                                 location=20, slices=3):
        raw = raw_data.getArray()
        distorted_raw = raw.copy()
        duplicate_slices = np.repeat(raw[location, :, :].reshape(1, raw.shape[1], raw.shape[2]), slices, axis=0)
        distorted_raw[location:location+slices, :, :] = duplicate_slices

        return distorted_raw, label
