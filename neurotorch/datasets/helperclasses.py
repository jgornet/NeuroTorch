import random
from torch.utils.data import Dataset
import numpy as np


class DatasetSplitter(object):
    """
    Balances a dataset with the given ratio split between the training and
    testing datasets
    """
    def __init__(self, dataset: Dataset, ratio=0.2):
        """
        Initializes the split dataset

        :param dataset: The PyTorch dataset to split
        :param ratio: The ratio of training to testing samples
        """
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
        Retrieves the training dataset from the DatasetSplitter
        """
        return self.train_dataset

    def getTestDataset(self):
        """
        Retrieves the testing dataset from the DatasetSplitter
        """
        return self.test_dataset


class SampledDataset(Dataset):
    """
    Creates a subsampled dataset using the given indices
    """
    def __init__(self, dataset, index_list):
        """
        Initializes a subsampled dataset using the given indices

        :param dataset: A PyTorch dataset to subsample
        :param index_list: A list of indexes in the PyTorch dataset to sample
        """
        self.index_list = index_list
        self.dataset = dataset

    def __len__(self):
        """
        Returns the length of the subsampled dataset
        """
        return len(self.index_list)

    def __getitem__(self, idx):
        """
        Returns a sample given by the index in the index list

        :param idx: An index bounded by the subsampled dataset size
        """
        return self.dataset[self.index_list[idx]]


class SupervisedDataset(Dataset):
    """
    Creates a input/label dataset for supervised training
    """
    def __init__(self, input_dataset, label_dataset):
        """
        Initializes the supervised dataset

        :param input_dataset: A PyTorch dataset containing inputs
        :param label_dataset: A PyTorch dataset containing corresponding labels
        """
        if len(input_dataset) != len(label_dataset):
            raise ValueError("Dataset sizes must be equal")

        if len(input_dataset) <= 0:
            raise ValueError("Dataset must have elements")

        self.input_dataset = input_dataset
        self.label_dataset = label_dataset

        self.length = len(input_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {"input": self.input_dataset[idx].astype(np.float32),
                  "label": self.label_dataset[idx].astype(np.float32)}

        return sample
