import tensorflow as tf
import numpy as np


class Trainer(object):
    """
    Trains a Tensorflow neural network with a given input and label dataset
    """
    def __init__(self, net, aligned_volume, checkpoint=None,
                 optimizer=None, criterion=None, max_epochs=10,
                 gpu_device=None, validation_split=0.2):
        """
        Sets up the parameters for training

        :param net: A TensorFlow neural network
        :param inputs_volume: A dataset containing inputs
        :param labels_volume: A dataset containing corresponding labels
        """
        self.max_epochs = max_epochs

        self.net = net

        if checkpoint is not None:
            self.checkpoint = checkpoint

        if optimizer is None:
            self.optimizer = tf.train.AdamOptimizer()
        else:
            self.optimizer = optimizer()

        if criterion is None:
            self.criterion = tf.losses.sigmoid_cross_entropy
        else:
            self.criterion = criterion

        if gpu_device is not None:
            self.gpu_device = gpu_device
            self.useGpu = True

        self.volume = aligned_volume

    def getTrainer(self):
        return self

    def run_epoch(self, sample_batch):
        """
        Runs an epoch with a given batch of samples

        :param sample_batch: A dictionary containing inputs and labels with the keys 
"input" and "label", respectively
        """
        with tf.GradientTape() as tape:
            inputs = tf.convert_to_tensor(np.expand_dims(sample_batch[0],
                                                         axis=1),
                                          dtype=tf.float32)
            labels = tf.convert_to_tensor(np.expand_dims(sample_batch[1],
                                                         axis=1),
                                          dtype=tf.float32)

            outputs = self.net(inputs)

            loss = self.criterion(labels, outputs)

        variables = self.net.variables
        gradients = tape.gradient(loss, self.net.variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def evaluate(self, batch):
        inputs = tf.convert_to_tensor(np.expand_dims(batch[0],
                                                        axis=1),
                                        dtype=tf.float32)
        labels = tf.convert_to_tensor(np.expand_dims(batch[1],
                                                        axis=1),
                                        dtype=tf.float32)

        outputs = self.net(inputs)

        loss = self.criterion(labels, outputs)
        accuracy = tf.math.reduce_sum(tf.cast((outputs > 0) & (labels > 0), tf.int32))
        accuracy /= tf.math.reduce_sum(tf.cast((outputs > 0) | (labels > 0), tf.int32))

        return loss, accuracy, tf.stack(outputs)

    def run_training(self):
        """
        Trains the given neural network
        """
        num_epoch = 1
        num_iter = 1

        validation_split = 0.2
        valid_indexes = self.getTrainer().volume.getValidData()
        random_idx = np.random.permutation(valid_indexes)
        train_idx = random_idx[:int(len(valid_indexes)*(1-validation_split))].copy()
        val_idx = random_idx[int(len(valid_indexes)*validation_split):].copy()

        train_idx = train_idx[:(len(train_idx) - len(train_idx) % 16)]
        train_idx = train_idx.reshape((-1, 16))

        val_idx = val_idx[:(len(val_idx) - len(val_idx) % 16)]
        val_idx = val_idx.reshape((-1, 16))

        while num_epoch <= self.getTrainer().max_epochs:
            np.random.shuffle(train_idx)
            for i in range(train_idx.shape[0]):
                sample_batch = list(zip(*[self.getTrainer().volume[idx]
                                          for idx in train_idx[i]]))
                sample_batch = [np.stack([x.getArray() for x in batch]) for batch in sample_batch]
                sample_batch[1] = sample_batch[1] > 0
                if num_epoch > self.getTrainer().max_epochs:
                    break
                if (sample_batch[1] == 0).all():
                    continue

                print("Iteration: {}".format(num_iter))
                self.run_epoch(sample_batch)

                if num_iter % 10 == 0:
                    val_batch = list(zip(*[self.getTrainer().volume[idx]
                                           for idx in val_idx[1]]))
                    val_batch = [np.stack([x.getArray() for x in batch]) for batch in val_batch]
                    val_batch[1] = val_batch[1] > 0
                    loss, accuracy, _ = self.evaluate(val_batch)
                    print("Iteration: {}".format(num_iter),
                          "Epoch {}/{} ".format(num_epoch,
                                                self.getTrainer().max_epochs),
                          "Loss: {:.4f}".format(loss),
                          "Accuracy: {:.2f}".format(accuracy*100))

                num_iter += 1

            num_epoch += 1


class TrainerDecorator(Trainer):
    """
    A wrapper class to a features for training
    """
    def __init__(self, trainer):
        self.setTrainer(trainer)

    def setTrainer(self, trainer):
        self._trainer = trainer

    def getTrainer(self):
        if isinstance(self._trainer, TrainerDecorator) or issubclass(type(self._trainer), TrainerDecorator):
            return self._trainer.getTrainer()
        if isinstance(self._trainer, Trainer):
            return self._trainer

    def run_epoch(self, sample_batch):
        return self._trainer.run_epoch(sample_batch)

    def evaluate(self, batch):
        return self._trainer.evaluate(batch)

    def run_training(self):
        """
        Trains the given neural network
        """
        num_epoch = 1
        num_iter = 1

        validation_split = 0.2
        valid_indexes = self.getTrainer().volume.getValidData()
        random_idx = np.random.permutation(valid_indexes)
        train_idx = random_idx[:int(len(valid_indexes)*(1-validation_split))].copy()
        val_idx = random_idx[int(len(valid_indexes)*validation_split):].copy()

        train_idx = train_idx[:(len(train_idx) - len(train_idx) % 16)]
        train_idx = train_idx.reshape((-1, 16))

        val_idx = val_idx[:(len(val_idx) - len(val_idx) % 16)]
        val_idx = val_idx.reshape((-1, 16))

        while num_epoch <= self.getTrainer().max_epochs:
            np.random.shuffle(train_idx)
            for i in range(train_idx.shape[0]):
                sample_batch = list(zip(*[self.getTrainer().volume[idx]
                                          for idx in train_idx[i]]))
                sample_batch = [np.stack([x.getArray() for x in batch]) for batch in sample_batch]
                sample_batch[1] = sample_batch[1] > 0
                if num_epoch > self.getTrainer().max_epochs:
                    break
                if (sample_batch[1] == 0).all():
                    continue

                print("Iteration: {}".format(num_iter))
                self.run_epoch(sample_batch)

                if num_iter % 10 == 0:
                    val_batch = list(zip(*[self.getTrainer().volume[idx]
                                           for idx in val_idx[1]]))
                    val_batch = [np.stack([x.getArray() for x in batch]) for batch in val_batch]
                    val_batch[1] = val_batch[1] > 0
                    loss, accuracy, _ = self.evaluate(val_batch)
                    print("Iteration: {}".format(num_iter),
                          "Epoch {}/{} ".format(num_epoch,
                                                self.getTrainer().max_epochs),
                          "Loss: {:.4f}".format(loss),
                          "Accuracy: {:.2f}".format(accuracy*100))

                num_iter += 1

            num_epoch += 1
