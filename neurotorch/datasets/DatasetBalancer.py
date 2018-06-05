import random
from torch.utils.data import Dataset
import numpy as np


class DatasetBalancer:
    """
    Balances a dataset with the given ratio split between the training and
    testing datasets
    """
    def __init__(self, dataset, ratio=0.2):
        self.dataset = dataset
        self.ratio = ratio

        self._splitDataset()

    def _splitDataset(self):
        """
        Split the dataset into a training and testing dataset based on the
        ratio
        """
        # Creates a set of indexes from the dataset and randomly divides the
        # indices into the training and testing samples with a distribution
        # given by the ratio
        samples = range(len(self.dataset))
        test_samples = list(random.sample(samples,
                                          int(self.ratio*len(self.dataset))))
        train_samples = np.setdiff1d(samples, test_samples).tolist()

        # Creates the new balanced datasets with the sample distributions
        self.train_dataset = SampledDataset(self.dataset, train_samples)
        self.test_dataset = SampledDataset(self.dataset, test_samples)

    def getTrainDataset(self):
        """
        Retrieves the training dataset from the DatasetBalancer
        """
        return self.train_dataset

    def getTestDataset(self):
        """
        Retrieves the testing dataset from the DatasetBalancer
        """
        return self.test_dataset


class SampledDataset(Dataset):
    """
    Creates a subsampled dataset using the indexes from samples
    """
    def __init__(self, dataset, samples):
        self.samples = samples
        self.dataset = dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.dataset[self.samples[idx]]
