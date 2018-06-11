from neurotorch.core.Trainer import TrainerDecorator
import torch
import os.path


class CheckpointWriter(TrainerDecorator):
    def __init__(self, trainer, checkpoint_dir, checkpoint_period=5000):
        if not os.path.isdir(checkpoint_dir):
            raise IOError("{} is not a valid directory".format(checkpoint_dir))

        super().__init__(trainer)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_period = checkpoint_period

    def run_epoch(self, sample_batch):
        iteration = super().__init__(sample_batch)

        if iteration["iteration"] % self.checkpoint_period == 0:
            self.save_checkpoint()

    def save_checkpoint(self, iteration):
        checkpoint_filename = os.path.join(self.checkpoint_dir,
                                           "iteration_{}.ckpt".format(iteration))
        torch.save(self.net.state_dict(), checkpoint_filename)
