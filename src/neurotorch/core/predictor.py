import torch
from torch.autograd import Variable
import numpy as np
from neurotorch.datasets.dataset import Data
import tensorflow as tf


class Predictor:
    """
    A predictor segments an input volume into an output volume
    """
    def __init__(self, net, checkpoint, gpu_device=None):
        self.setNet(net, gpu_device=gpu_device)
        self.checkpoint = tf.train.Checkpoint(net=net)
        self.loadCheckpoint(checkpoint)

    def setNet(self, net, gpu_device=None):
        self.net = net

    def getNet(self):
        return self.net

    def loadCheckpoint(self, checkpoint):
        self.checkpoint.restore(checkpoint)

    def run(self, input_volume, output_volume, batch_size=20):
        self.setBatchSize(batch_size)

        batch_list = [list(range(len(input_volume)))[i:i+self.getBatchSize()]
                        for i in range(0,
                                        len(input_volume),
                                        self.getBatchSize())]

        for batch_index in batch_list:
            batch = [input_volume[i] for i in batch_index]

            self.run_batch(batch, output_volume)

    def getBatchSize(self):
        return self.batch_size

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

    def run_batch(self, batch, output_volume):
        bounding_boxes, arrays = self.toTensor(batch)
        inputs = arrays

        outputs = self.getNet()(inputs)

        data_list = self.toData(outputs, bounding_boxes)
        for data in data_list:
            output_volume.blend(data)

    def toArray(self, data):
        torch_data = data.getArray().astype(np.float)
        torch_data = torch_data.reshape(1, 1, *torch_data.shape)
        return torch_data

    def toTensor(self, batch):
        bounding_boxes = [data.getBoundingBox() for data in batch]
        arrays = [self.toArray(data) for data in batch]
        arrays = tf.convert_to_tensor(np.concatenate(arrays, axis=0),
                                      dtype=tf.float32)

        return bounding_boxes, arrays

    def toData(self, tensor_list, bounding_boxes):
        tensor = tensor_list.numpy()
        batch = [Data(tensor[i][0], bounding_box)
                 for i, bounding_box in enumerate(bounding_boxes)]

        return batch
