from neurotorch.datasets.dataset import AlignedVolume
from neurotorch.datasets.datatypes import Vector
from abc import abstractmethod
import numpy as np
from random import random


class Augmentation(AlignedVolume):
    def __init__(self, aligned_volume, iteration_size=None, stride=None,
                 frequency=1.0):
        self.setFrequency(frequency)
        self.setVolume(aligned_volume)
        self.setAugmentation(True)
        if iteration_size is None:
            iteration_size = self.getInputVolume().getIterationSize()
        if stride is None:
            stride = self.getInputVolume().getStride()
        self.setIteration(iteration_size, stride)

    def setAugmentation(self, augment):
        self.eval = augment

    def getAugmentation(self):
        return self.eval

    def get(self, bounding_box):
        if random() < self.frequency and self.getAugmentation():
            augmented_data = self.augment(bounding_box)
            return augmented_data
        else:
            data = (self.getInput(bounding_box), self.getLabel(bounding_box))
            return data

    def setFrequency(self, frequency=1.0):
        self.frequency = frequency

    def getBoundingBox(self):
        return self.getVolume().getBoundingBox()

    def setIteration(self, iteration_size, stride):
        self.getParent().setIterationSize(iteration_size)
        self.getParent().setStride(stride)

    def getInputVolume(self):
        return self.getVolume().getVolumes()[0]

    def getLabelVolume(self):
        return self.getVolume().getVolumes()[1]

    def getInput(self, bounding_box):
        return self.getInputVolume().get(bounding_box)

    def getLabel(self, bounding_box):
        return self.getLabelVolume().get(bounding_box)

    def setIterationSize(self, iteration_size):
        self.iteration_size = iteration_size

    def setStride(self, stride):
        self.stride = stride

    def setVolume(self, aligned_volume):
        self.aligned_volume = aligned_volume

    def getVolume(self):
        if isinstance(self.aligned_volume, AlignedVolume):
            return self.aligned_volume 
        if isinstance(self.aligned_volume, Augmentation) or \
           issubclass(type(self.aligned_volume), Augmentation):
            return self.aligned_volume.getVolume()

    def getVolumes(self):
        return self.getVolume().getVolumes()

    def getParent(self):
        return self.aligned_volume

    def __len__(self):
        return len(self.getParent())

    def __getitem__(self, idx):
        bounding_box = self.getParent()._indexToBoundingBox(idx)
        result = self.get(bounding_box)

        return result

    @abstractmethod
    def augment(self, bounding_box):
        pass

    def getValidData(self):
        return self.getParent().getValidData()
