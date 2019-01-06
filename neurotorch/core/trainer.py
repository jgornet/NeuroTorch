import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from neurotorch.datasets.dataset import AlignedVolume, TorchVolume
import torch.cuda
import numpy as np


class Trainer(object):
    """
    Trains a PyTorch neural network with a given input and label dataset
    """
    def __init__(self, net, inputs_volume, labels_volume, checkpoint=None,
                 optimizer=None, criterion=None, max_epochs=10,
                 gpu_device=None, validation_split=0.2):
        """
        Sets up the parameters for training

        :param net: A PyTorch neural network
        :param inputs_volume: A PyTorch dataset containing inputs
        :param labels_volume: A PyTorch dataset containing corresponding labels
        """
        self.max_epochs = max_epochs

        self.device = torch.device("cuda:{}".format(gpu_device)
                                   if gpu_device is not None
                                   else "cpu")

        self.net = net.to(self.device)

        if checkpoint is not None:
            self.net.load_state_dict(torch.load(checkpoint))

        if optimizer is None:
            self.optimizer = optim.Adam(self.net.parameters())
        else:
            self.optimizer = optimizer

        if criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = criterion

        if gpu_device is not None:
            self.gpu_device = gpu_device
            self.useGpu = True

        self.volume = TorchVolume(AlignedVolume((inputs_volume,
                                                 labels_volume)))

        random_idx = np.random.choice(len(self.volume))
        train_idx = random_idx[:int(len(self.volume)*(1-validation_split))]
        val_idx = random_idx[int(len(self.volume)*validation_split):]

        self.training_data = DataLoader(self.volume, sampler=train_idx,
                                        batch_size=16, num_workers=4,
                                        drop_last=True)

        self.validation_data = DataLoader(self.volume, sampler=val_idx,
                                          batch_size=16, num_workers=1,
                                          drop_last=True)

    def run_epoch(self, sample_batch):
        """
        Runs an epoch with a given batch of samples

        :param sample_batch: A dictionary containing inputs and labels with the keys 
"input" and "label", respectively
        """
        inputs = Variable(sample_batch[0]).float()
        labels = Variable(sample_batch[1]).float()

        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.net(inputs)

        loss = self.criterion(torch.cat(outputs), labels)
        loss_hist = loss.cpu().item()
        loss.backward()
        self.optimizer.step()

        return loss_hist

    def evaluate(self, batch):
        inputs = Variable(sample_batch[0]).float()
        labels = Variable(sample_batch[1]).float()

        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.net(inputs)

        loss = self.criterion(torch.cat(outputs), labels)
        accuracy = sum((torch.cat(outputs) > 0) & labels) / \
                   sum((torch.cat(outputs) > 0) | labels)

        return loss.cpu().item(), accuracy.cpu().item(), outputs.cpu().numpy()

    def run_training(self):
        """
        Trains the given neural network
        """
        num_epoch = 1
        num_iter = 1
        while num_epoch <= self.max_epochs:
            for i, sample_batch in enumerate(self.training_data):
                if num_epoch > self.max_epochs:
                    break
                if (sum(sample_batch[1]) / sample_batch[1].numel()) < 0.25:
                    continue

                self.run_epoch(sample_batch)

                if num_iter % 100:
                    loss, accuracy, _ = self.evaluate(next(self.validation_data))
                    print("Iteration: {}".format(num_iter),
                          "Epoch {}/{} ".format(num_epoch,
                                                self.max_epochs),
                          "Loss: {:.4f}".format(loss),
                          "Accuracy: {:.2f}".format(accuracy*100))

                num_iter += 1

            num_epoch += 1


class TrainerDecorator(Trainer):
    """
    A wrapper class to a features for training
    """
    def __init__(self, trainer):
        if isinstance(trainer, TrainerDecorator):
            self._trainer = trainer._trainer
        if isinstance(trainer, Trainer):
            self._trainer = trainer
        else:
            error_string = ("trainer must be a Trainer or TrainerDecorator " +
                            "instead it has type {}".format(type(trainer)))
            raise ValueError(error_string)

    def run_epoch(self, sample_batch):
        return self._trainer.run_epoch(sample_batch)

    def run_training(self):
        num_epoch = 1
        while num_epoch <= self._trainer.max_epochs:
            for i, sample_batch in enumerate(self._trainer.data_loader):
                if num_epoch > self._trainer.max_epochs:
                    break
                loss = self.run_epoch(sample_batch)
                print("Epoch {}/{}".format(num_epoch,
                                           self._trainer.max_epochs),
                      "Loss: {:.4f}".format(loss))
                num_epoch += 1
