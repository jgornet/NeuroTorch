from neurotorch.augmentations.augmentation import Augmentation
from neurotorch.datasets.dataset import Data
from neurotorch.datasets.datatypes import Vector, BoundingBox
import random
import numpy as np
from scipy.ndimage import zoom


class Drop(Augmentation):
    def __init__(self, volume, max_slices=20, **kwargs):
        self.setMaxDroppedSlices(max_slices)
        super().__init__(volume, **kwargs)

    def augment(self, bounding_box):
        # Get dropped slices and location
        dropped_slices = 2*random.randrange(1, self.max_slices//2)
        location = random.randrange(dropped_slices,
                                    bounding_box.getSize()[1]-dropped_slices)

        # Get enlarged bounding box
        edge1, edge2 = bounding_box.getEdges()
        edge2 += Vector(0, dropped_slices, 0)
        initial_bounding_box = BoundingBox(edge1, edge2)

        # Get data
        raw, label = self.getParent().get(initial_bounding_box)

        # Augment Numpy arrays
        augmented_raw, augmented_label = self.drop(raw, label,
                                                   dropped_slices=dropped_slices,
                                                   location=location)

        # Convert back into the data format
        augmented_raw_data = Data(augmented_raw, bounding_box)
        augmented_label_data = Data(augmented_label, bounding_box)

        return (augmented_raw_data, augmented_label_data)

    def setMaxDroppedSlices(self, max_slices):
        self.max_slices = max_slices

    def setLocation(self, location):
        self.location = location

    def drop(self, raw_data, label_data, dropped_slices=1, location=0):
        # Initialize distorted raw volume and label
        raw = raw_data.getArray()
        label = label_data.getArray()
        distorted_raw = np.zeros((raw.shape[0], raw.shape[1]-dropped_slices, raw.shape[2]))
        distorted_label = np.zeros((label.shape[0], label.shape[1]-dropped_slices, label.shape[2]))

        # Populate distorted raw volume and label
        distorted_raw[:, :location-dropped_slices//2, :] = raw[:, :location-dropped_slices//2, :]
        distorted_raw[:, location-dropped_slices//2:, :] = raw[:, location+dropped_slices//2:, :]

        distorted_label[:, :location-dropped_slices//2, :] = label[:, :location-dropped_slices//2, :]
        distorted_label[:, location-dropped_slices//2:, :] = label[:, location+dropped_slices//2:, :]

        # Interpolate the distorted label
        mag = 0.5
        fill_region = label[:, location-dropped_slices:location+dropped_slices, :]
        fill_region = zoom(fill_region, zoom=(1, mag, 1))
        fill_region = (fill_region > 0)

        # Fill in distorted label with interpolation
        distorted_label[:, location-dropped_slices:location, :] = fill_region

        return distorted_raw.astype(raw.dtype), distorted_label.astype(label.dtype)
