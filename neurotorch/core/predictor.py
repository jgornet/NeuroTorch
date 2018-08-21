import torch
from torch.autograd import Variable
import numpy as np
from neurotorch.datasets.dataset import Data


class Predictor(object):
    def __init__(self, net, checkpoint, gpu_device=None):
        self.setNet(net, gpu_device=gpu_device)
        self.loadCheckpoint(checkpoint)

    def setNet(self, net, gpu_device=None):
        self.device = torch.device("cuda:{}".format(gpu_device)
                                   if gpu_device is not None
                                   else "cpu")

        self.net = net.to(self.device)

    def getNet(self):
        return self.net

    def loadCheckpoint(self, checkpoint):
        self.getNet().load_state_dict(torch.load(checkpoint))

    def run(self, input_volume, output_volume, batch_size=8):
        self.setBatchSize(batch_size)

        with torch.no_grad():
            self.run_batch(self.getBatch(input_volume), output_volume)

    def getBatch(self, input_volume):
        batch = []
        for i in range(self.getBatchSize()):
            try:
                batch.append(next(input_volume))
            except StopIteration:
                break

        return batch

    def getBatchSize(self):
        return self.batch_size

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

    def run_batch(self, batch, output_volume):
        # if len(self.getBatch()) != self.getBatchSize():
        #     raise ValueError("Batch number must equal set batch size")

        bounding_boxes, arrays = self.toTorch(batch)
        inputs = Variable(arrays).float()

        outputs = self.getNet()(inputs)
        data_list = self.toData(outputs, bounding_boxes)
        for data in data_list:
            output_volume.blend(data)
            if (data.getArray() == 0.0).any():
                print(data.getArray())

    def toArray(self, data):
        torch_data = data.getArray().astype(np.float)
        torch_data = torch_data.reshape(1, 1, *torch_data.shape)
        return torch_data

    def toTorch(self, batch):
        bounding_boxes = [data.getBoundingBox() for data in batch]
        arrays = [self.toArray(data) for data in batch]
        arrays = torch.from_numpy(np.concatenate(arrays, axis=0))
        arrays = arrays.to(self.device)

        return bounding_boxes, arrays

    def toData(self, tensor_list, bounding_boxes):
        batch = [Data(tensor.cpu().numpy()[0], bounding_box)
                 for tensor, bounding_box in zip(tensor_list, bounding_boxes)]

        return batch
