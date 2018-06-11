from neurotorch.core.Trainer import TrainerDecorator
import tensorboardX
import os.path
import logging


class LossWriter(TrainerDecorator):
    def __init__(self, trainer, logger_dir):
        if not os.path.isdir(logger_dir):
            raise IOError("{} is not a valid directory".format(logger_dir))

        super().__init__(trainer)
        self.train_writer = tensorboardX.SummaryWriter(logger_dir +
                                                       "train_log")
        self.validation_writer = tensorboardX.SummaryWriter(logger_dir +
                                                            "validation_log")

    def log_loss(self, loss, time, iteration):
        self.train_writer.add_scalar("Time", time, iteration)
        self.train_writer.add_scalar("Loss", loss, iteration)

    def run_epoch(self, sample_batch):
        iteration = super().run_epoch(sample_batch)

        self.log_loss(iteration["loss"], iteration["time"],
                      iteration["iteration"])


class TrainingLogger(TrainerDecorator):
    def __init__(self, trainer, logger_dir=None):
        if not os.path.isdir(logger_dir) and logger_dir is not None:
            raise IOError("{} is not a valid directory".format(logger_dir))

        super().__init__(trainer)

        if logger_dir:
            logging.basicConfig(filename=os.path.join(logger_dir,
                                                      'training.log'),
                                level=logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO)

    def run_epoch(self, sample_batch):
        iteration = super().run_epoch(sample_batch)

        logging.info("Iteration: {}, Loss: {}".format(iteration["iteration"],
                                                      iteration["loss"]))


class ImageWriter(TrainerDecorator):
    def __init__(self, trainer, image_dir):
        if not os.path.isdir(image_dir):
            raise IOError("{} is not a valid directory".format(image_dir))
