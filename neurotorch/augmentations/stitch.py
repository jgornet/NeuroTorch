from neurotorch.augmentations.augmentation import Augmentation
from neurotorch.datasets.dataset import Data
import random
import numpy as np


class Stitch(Augmentation):
    def __init__(self, volume, max_error=2, **kwargs):
        self.setMaxError(max_error)
        super().__init__(volume, **kwargs)

    def setMaxError(self, max_error):
        self.max_error = max_error

    def augment(self, bounding_box):
        # Get error and location
        error = random.randrange(self.max_error)
        x_len = bounding_box.getSize()[0]
        location = random.randrange(10, x_len - 10)

        # Get initial bounding box
        edge1, edge2 = bounding_box.getEdges()
        edge2 += Vector(20, error, 0)
        initial_bounding_box = BoundingBox(edge1, edge2)

        # Get data
        raw_data = self.getInput(initial_bounding_box).getArray()
        label_data = self.getLabel(initial_bounding_box).getArray()
        augmented_raw, augmented_label = self.duplication(raw_data, label_data,
                                                          location=location,
                                                          error=error)

        # Convert to the data format
        augmented_raw_data = Data(augmented_raw, bounding_box)
        augmented_label_data = Data(augmented_label, bounding_box)

        return (augmented_raw_data, augmented_label_data)

    def stitch(self, raw, label, location=20, error=3)
        # Initialize distorted raw volume and label
        z_len, y_len, x_len = raw.shape

        distorted_raw = np.zeros((z_len,
                                y_len-error,
                                x_len)).astype(raw.dtype)

        distorted_label = np.zeros((z_len,
                                    y_len-error,
                                    x_len)).astype(label.dtype)

        # Shear raw volume
        distorted_raw[:, :, location:] = raw[:, :-error, location:]
        distorted_raw[:, :, :location] = raw[:, error:, :location]

        distorted_label[:, :, location:] = label[:, :-error, location:]
        distorted_label[:, :, :location] = label[:, error:, :location]

        # Shear label
        fill_region = label[:, error//2:-error//2,
                            location-10:location+10]
        fill_region = shear3d(fill_region, shear=error, axis=1)
        distorted_label[:, :, location-10:location+10] = fill_region


        return augmented_raw_data, label_data

    def shear3d(self, volume, shear=20, axis=1):
        result = volume.copy()
        shift_list = np.around(np.linspace(-shear//2, shear//2, num=result.shape[2])).astype(np.int8)
        for index, shift in enumerate(shift_list):
            result[:, :, index] = np.roll(result[:, :, index], shift, axis=axis)

        return result
