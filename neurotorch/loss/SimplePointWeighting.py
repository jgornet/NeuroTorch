import torch
from torch.nn import BCEWithLogitsLoss, Module
import numpy as np
from scipy.ndimage.measurements import label


class SimplePointBCEWithLogitsLoss(Module):
    """
    Weights the binomial cross-entropy loss by the non-simple points
    """

    def __init__(self, simple_weight=1, non_simple_weight=1):
        super().__init__()
        self.simple_weight = simple_weight
        self.non_simple_weight = non_simple_weight
        self.bce = BCEWithLogitsLoss(reduction='none')

    def forward(self, prediction, label):
        simple_weight = self.simple_weight
        non_simple_weight = self.non_simple_weight

        prediction_weights = self.simple_weight(
            prediction, simple_weight=0, non_simple_weight=1,
        )
        label_weights = self.simple_weight(
            label, simple_weight=0, non_simple_weight=1,
        )

        weight = (prediction_weights + label_weights) > 0
        weight = (weight.float() * non_simple_weight) + \
            ((~weight).float() * simple_weight)

        cost = self.bce(prediction, label)
        cost = weight * cost

        return cost.mean()

    def simple_weight(self, tensor, simple_weight=1, non_simple_weight=1):
        non_simple_points = self.label_nonsimple_points(tensor)
        simple_points = tensor.new_ones(tensor.size()).to(tensor.get_device()) - \
            non_simple_points
        inputs_weights = non_simple_weight * non_simple_points + \
            simple_weight * simple_points
        return inputs_weights

    def label_nonsimple_points(self, tensor, threshold=0):
        """
        Labels every non-simple point in a tensor

        :param tensor: A PyTorch tensor
        :param threshold: The threshold to binarize the tensor
        """
        try:
            device = tensor.get_device()
        except RuntimeError:
            raise RuntimeError("simple point weighting currently only works" +
                               " for GPUs")
        array = tensor.to("cpu")
        array = array.data.numpy()
        array = (array > threshold)
        labeled_array, num_features = label(array)
        size = labeled_array.shape
        padded_array = np.pad(labeled_array, (1,), 'edge')
        result = np.zeros(size)

        for k in range(0, size[0]):
            for j in range(0, size[1]):
                for i in range(0, size[2]):
                    if self._is_nonsimple_point(padded_array[k:k+3,
                                                             j:j+3,
                                                             i:i+3]):
                        result[k, j, i] = 1

        result = torch.from_numpy(result).to(device).type(type(tensor))

        return result

    def _is_nonsimple_point(self, neighborhood):
        """
        Determines whether the center voxel in a labeled 3x3 neighborhood is simple

        :param neighborhood: A labeled 3x3 Numpy array
        """
        # Skip if the point is background
        if (neighborhood[1, 1, 1] == 0).any():
            return False

        # Setup neighborhood
        result = np.copy(neighborhood)
        center_point_label = result[1, 1, 1]

        # Create 18-neighborhood structure
        s = np.zeros((3, 3, 3))
        s[0, :, :] = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        s[1, :, :] = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]])
        s[2, :, :] = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])

        # Calculates the topological number of the cavity
        result[result == 0] = -1
        labeled_array, num_features = label(result != center_point_label,
                                            structure=s)

        if num_features != 1:
            return True

        # Calculates the topological number of the component
        result = (result == center_point_label)
        result[1, 1, 1] = 0
        labeled_array, num_features = label(result,
                                            structure=np.ones((3, 3, 3)))

        if num_features != 1:
            return True

        # If the prior conditions are not satisfied, the point is simple
        return False
