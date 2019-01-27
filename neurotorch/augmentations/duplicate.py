from neurotorch.augmentations.augmentation import Augmentation
from neurotorch.datasets.dataset import Data
import random
import numpy as np
from scipy.ndimage.filters import convolve


class Duplicate(Augmentation):
    def __init__(self, volume, max_slices=2, **kwargs):
        self.setMaxSlices(max_slices)
        super().__init__(volume, **kwargs)

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

    def setMaxSlices(self, max_slices):
        self.max_slices = max_slices

    def getMaxSlices(self):
        return self.max_slices

    def getSlices(self):
        return random.randrange(self.getMaxSlices())

    def duplication(self, raw_data, label_data, location=20, slices=3,
                    axis=0):
        raw = raw_data.getArray()
        distorted_raw = raw.copy()

        noise = raw[:, :, location:location+slices]
        noise = noise - convolve(noise, weights=np.full((3, 3, 3), 1.0/27))

        duplicate_slices = np.repeat(raw[:, :, location+slices//2].reshape(raw.shape[0],
                                                                           raw.shape[1],
                                                                           1),
                                     slices, axis=2)
        duplicate_slices += noise
        distorted_raw[:, :, location:location+slices] = duplicate_slices

        augmented_raw_data = Data(distorted_raw, raw_data.getBoundingBox())

        return augmented_raw_data, label_data
