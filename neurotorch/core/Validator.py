import torch
from neurotorch.datasets.stitcher import TiffStitcher
from neurotorch.datasets.volumedataset import ThreeDimDataset
from torch.autograd import Variable


class Validator(TiffStitcher):
    def __init__(self, net, dataset: ThreeDimDataset, checkpoint,
                 gpu_device=None):
        super().__init__(self, dataset)
        self.device = torch.device("cuda:{}".format(gpu_device)
                                   if gpu_device is not None else "cpu")
        self.net = net.to(self.device)
        self.dataset = dataset

        if checkpoint is None:
            raise ValueError("A training checkpoint must be given")
        else:
            self.net.load_state_dict(torch.load(checkpoint))

    def add_sample(self, sample, index):
        inputs = Variable(sample)
        inputs = inputs.to(self.device)
        outputs = self.net(inputs)

        super().add_sample(outputs.numpy(), index)
