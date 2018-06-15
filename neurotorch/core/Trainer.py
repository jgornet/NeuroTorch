import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from neurotorch.datasets.DatasetBalancer import (DatasetSplitter,
                                                 SupervisedDataset)
import torch.cuda


class Trainer(object):
    def __init__(self, net, inputs_dataset, labels_dataset, checkpoint=None,
                 optimizer=None, criterion=None, max_epochs=100000,
                 gpu_device=None):
        self.max_epochs = max_epochs

        self.device = torch.device("cuda:{}".format(gpu_device)
                                   if gpu_device is not None
                                   else "cpu")

        self.net = net.to(self.device)

        if checkpoint is not None:
            self.net.load_state_dict(torch.load(checkpoint))

        if optimizer is None:
            self.optimizer = optim.Adam(self.net.parameters())

        if criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()

        if gpu_device is not None:
            self.gpu_device = gpu_device
            self.useGpu = True

        self.split_dataset = DatasetSplitter(SupervisedDataset(inputs_dataset,
                                                               labels_dataset))

        self.data_loader = DataLoader(self.split_dataset.getTrainDataset(),
                                      batch_size=1, shuffle=True,
                                      num_workers=1)

    def run_epoch(self, sample_batch):
        inputs = sample_batch["input"]
        labels = sample_batch["label"]

        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.net(Variable(inputs.float()))
        loss = self.criterion(*outputs,
                              Variable(labels.float()))
        loss.backward()
        self.optimizer.step()

        return loss

    def run_training(self):
        num_epoch = 1
        while num_epoch <= self.max_epochs:
            for i, sample_batch in enumerate(self.data_loader):
                if num_epoch > self.max_epochs:
                    break
                print("Epoch {}/{}".format(num_epoch,
                                            self.max_epochs))
                self.run_epoch(sample_batch)
                num_epoch += 1


class TrainerDecorator(Trainer):
    def __init__(self, trainer):
        self._trainer = trainer

    def run_epoch(self, sample_batch):
        return self._trainer.run_epoch(sample_batch)

    def run_training(self):
        num_epoch = 1
        while num_epoch <= self._trainer.max_epochs:
            for i, sample_batch in enumerate(self._trainer.data_loader):
                if num_epoch > self._trainer.max_epochs:
                    break
                print("Epoch {}/{}".format(num_epoch,
                                            self._trainer.max_epochs))
                self.run_epoch(sample_batch)
                num_epoch += 1
