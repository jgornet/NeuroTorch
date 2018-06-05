import torch


class Trainer:
    def __init__(self, net, checkpoint=None):
        self.net = torch.nn.DataParallel(net).cuda()

        if checkpoint is not None:
            self.net.load_state_dict(torch.load(checkpoint))

    def train(self):
        
