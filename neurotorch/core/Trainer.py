import torch
import torch.optim as optim
import torch.nn as nn


class Trainer(object):
    def __init__(self, net, data_loader, checkpoint=None,
                 optimizer=None, criterion=None, max_epochs=100000):
        self.net = torch.nn.DataParallel(net).cuda()

        if checkpoint is not None:
            self.net.load_state_dict(torch.load(checkpoint))

        if optimizer is None:
            optimizer = optim.Adam(net.parameters())

        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()

        self.data_loader = data_loader

    def run_epoch(self, sample_batch):
        inputs = sample_batch["input"]
        labels = sample_batch["label"]

        self.optimizer.zero_grad()

        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return {"inputs": inputs, "labels": labels,
                "outputs": outputs, "loss": loss}

    def run_training(self):
        num_epoch = 0
        while num_epoch < self.max_epochs:
            for i, sample_batch in enumerate(self.data_loader):
                self.run_epoch()


class TrainerDecorator(Trainer):
    def __init__(self, trainer):
        self._trainer = trainer

    def run_epoch(self, sample_batch):
        self._trainer.run_epoch(sample_batch)

    def run_training(self):
        self._trainer.run_training()
