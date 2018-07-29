from neurotorch.augmentations.augmentation import Augmentation
from neurotorch.datasets.volumedataset import Data
import random
import numpy as np


class Brightness(Augmentation):
    def __init__(self, volume, frequency=0.1, max_relative_brightness=0.05):
        self.setFrequency(frequency)
        self.setRelativeBrightness(max_relative_brightness)
        super().__init__(volume)

    def augment(self, bounding_box):
        data = self.getVolume().get(bounding_box)
        raw = data[0].getArray()
        label = data[1].getArray()
        augmented_raw, augmented_label = self.brightness_augmentation(raw, label)

        raw_data = Data(augmented_raw, bounding_box)
        label_data = Data(augmented_label, bounding_box)

        return (raw_data, label_data)

    def setFrequency(self, frequency):
        self.frequency = frequency

    def setRelativeBrightness(self, relative_brightness):
        self.relative_brightness = relative_brightness

    def brightness_augmentation(self, raw, label, maximum=0.05):
        augmented_raw = raw
        brightness = random.uniform(0, maximum)
        augmented_raw = augmented_raw + brightness
        augmented_raw = augmented_raw.astype(np.uint16)

        return augmented_raw, label
