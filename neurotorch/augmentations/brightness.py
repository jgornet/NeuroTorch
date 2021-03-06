from neurotorch.augmentations.augmentation import Augmentation
from neurotorch.datasets.dataset import Data
import random


class Brightness(Augmentation):
    def __init__(self, volume, max_relative_brightness=0.05, **kwargs):
        self.setRelativeBrightness(max_relative_brightness)
        super().__init__(volume, **kwargs)

    def augment(self, bounding_box):
        raw, label = self.getParent().get(bounding_box)
        augmented_raw, augmented_label = self.brightness_augmentation(raw,
                                                                      label)

        return (augmented_raw, augmented_label)

    def setFrequency(self, frequency):
        self.frequency = frequency

    def setRelativeBrightness(self, relative_brightness):
        self.relative_brightness = relative_brightness

    def brightness_augmentation(self, raw_data, label_data,
                                brightness_range=0.05):
        augmented_raw = raw_data.getArray()
        brightness = random.uniform(-brightness_range, brightness_range)
        augmented_raw = augmented_raw + brightness
        augmented_raw_data = Data(augmented_raw, raw_data.getBoundingBox())

        return augmented_raw_data, label_data
