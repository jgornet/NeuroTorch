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

        self.iteration = 1

    def run_epoch(self, sample_batch):
        iteration = super().run_epoch(sample_batch)

        if self.iteration % self.checkpoint_period == 0:
            self.save_checkpoint(self.iteration)

        self.iteration += 1

    def save_checkpoint(self, iteration):
        checkpoint_filename = os.path.join(self.checkpoint_dir,
                                           "iteration_{}.ckpt".format(iteration))
        torch.save(self._trainer.net.state_dict(), checkpoint_filename)
