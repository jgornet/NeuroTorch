from neurotorch.core.Trainer import TrainerDecorator
import tensorflow as tf
import os.path
import logging
import time


class LossWriter(TrainerDecorator):
    def __init__(self, trainer, logger_dir):
        if not os.path.isdir(logger_dir):
            raise IOError("{} is not a valid directory".format(logger_dir))

        super().__init__(trainer)
        self.train_writer = tf.contrib.summary.SummaryWriter(logger_dir +
                                                             "train_log")
        self.validation_writer = tf.contrib.summary.SummaryWriter(logger_dir +
                                                                  "validation_log")

        self.iteration = 0

    def log_loss(self, loss, duration, iteration):
        self.train_writer.add_scalar("Time", duration, iteration)
        self.train_writer.add_scalar("Loss", loss, iteration)

    def run_epoch(self, sample_batch):
        start = time.time()
        loss = super().run_epoch(sample_batch)
        end = time.time()

        duration = start - end

        self.iteration += 1

        self.log_loss(loss, duration, self.iteration)


class TrainingLogger(TrainerDecorator):
    def __init__(self, trainer, logger_dir=None):
        if logger_dir is not None and not os.path.isdir(logger_dir):
            raise IOError("{} is not a valid directory".format(logger_dir))

        super().__init__(trainer)

        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(logging.INFO)

        if logger_dir:
            file_handler = logging.FileHandler(os.path.join(logger_dir,
                                                            "training.log"))
            self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        self.logger.addHandler(console_handler)

        self.iteration = 0

    def run_epoch(self, sample_batch):
        loss = super().run_epoch(sample_batch)
        self.iteration += 1

        self.logger.info("Iteration: {}, Loss: {}".format(self.iteration,
                                                          loss))

    def run_training(self):
        super().run_training()


class ImageWriter(TrainerDecorator):
    def __init__(self, trainer, image_dir):
        if not os.path.isdir(image_dir):
            raise IOError("{} is not a valid directory".format(image_dir))
