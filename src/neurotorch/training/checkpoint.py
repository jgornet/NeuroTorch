from neurotorch.core.trainer import TrainerDecorator
import tensorflow as tf
import os.path


class CheckpointWriter(TrainerDecorator):
    """
    A Trainer wrapper class to save training checkpoints
    """
    def __init__(self, trainer, checkpoint_dir, checkpoint_period=5000):
        """
        Initializes the checkpoint directory to write checkpoints

        :param trainer: Trainer object that the class wraps
        :param logger_dir: The directory to save checkpoints
        :param checkpoint_period: The number of iterations between checkpoints
        """
        if not os.path.isdir(checkpoint_dir):
            raise IOError("{} is not a valid directory".format(checkpoint_dir))

        super().__init__(trainer)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_period = checkpoint_period
        self.checkpoint = tf.train.Checkpoint(net=self.getTrainer().net,
                                              optimizer=self.getTrainer().optimizer)
        self.max_accuracy = 0

        self.iteration = 1

    def run_epoch(self, sample_batch):
        """
        Runs an epoch and saves the checkpoint if there have been enough iterations

        :param sample_batch: A batch of input/label samples for training
        """
        loss = super().run_epoch(sample_batch)

        if self.iteration % self.checkpoint_period == 0:
            self.save_checkpoint("iteration_{}.ckpt".format(self.iteration))

        self.iteration += 1

        return loss

    def evaluate(self, batch):
        loss, accuracy, output = super().evaluate(batch)

        if accuracy > self.max_accuracy:
            self.max_accuracy = accuracy
            self.save_checkpoint("best.ckpt")

        return loss, accuracy, output

    def save_checkpoint(self, checkpoint_name):
        """
        Saves a training checkpoint
        """
        checkpoint_filename = os.path.join(self.checkpoint_dir,
                                           checkpoint_name)
        self.checkpoint.save(checkpoint_filename)
