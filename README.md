# <img alt="NeuroTorch" src="/docs/images/NeuroTorch%20Logo.svg" height="80">

**NeuroTorch** is a framework for reconstructing neuronal morphology from
optical microscopy images. It interfaces PyTorch with different
automated neuron tracing algorithms for fast, accurate, scalable
neuronal reconstructions. It uses deep learning to generate an initial
segmentation of neurons in optical microscopy images. This
segmentation is then traced using various automated neuron tracing
algorithms to convert the segmentation into an SWC file—the most
common neuronal morphology file format. NeuroTorch is designed with
scalability in mind and can handle teravoxel-sized images easily.

## Core features

NeuroTorch provides several features for reconstructing neurons from optical microscopy images:
  * A platform-independent volumetric data loader that can handle teravoxel-sized datasets
  * A training framework that can handle various neural network architectures and paradigms
  * A segmentation framework that can scale to teravoxel-sized datasets efficiently
  * An automated tracing platform for converting segmentations to neuronal morphology files

## Installation
NeuroTorch can be installed in many ways and used within a few
minutes. To install NeuroTorch through pip, make sure Python 3.5 is
installed and type in the command-line

``` shell
$ pip install neurotorch
```

To install NeuroTorch through Docker, make sure Docker is installed and type

``` shell
$ docker run -it gornet/neurotorch:v0.1.0
```

Otherwise, NeuroTorch can be installed manually by typing in the command-line

``` shell
$ python setup.py install
```

## Quick Start

To get started with NeuroTorch, create the file hello-neurotorch.py
and edit its contents to contain

``` python
from neurotorch.datasets.dataset import TiffVolume
from neurotorch.core.trainer import Trainer
from neurotorch.nets.RSUNet import RSUNet
from neurotorch.core.predictor import Predictor


def main():
    net = RSUNet()  # Initialize the U-Net architecture

    inputs = TiffVolume("inputs.tif")  # Create a volume containing inputs
    labels = TiffVolume("labels.tif")  # Create a volume containing labels

    trainer = Trainer(net, inputs, labels,  # Setup a network trainer
                      max_epochs=100, gpu_device=0)

    # Set the trainer to add a checkpoint every 50 epochs
    trainer = CheckpointWriter(trainer, checkpoint_dir='.', 
                               checkpoint_period=50)

    trainer.run_training()  # Start training

    outputs = TiffVolume("outputs.tif")  # Create a volume for predictions

    # Setup a predictor for computing outputs
    predictor = Predictor(net, checkpoint='./iteration_100.ckpt',
                          batch_size=5)

    predictor.run(inputs, outputs, batch_size=5)  # Run prediction

    outputs.save()  # Save outputs

if __name__ == '__main__':
    main()

```

Next, let us run this script by typing in the command-line

``` shell
$ python hello-neurotorch.py
```

If everything worked out, you should have a file “outputs.tif”—TIFF
file containing the network prediction—in your directory.

## About

This project was created by James Gornet. Significant features were
rewritten from code provided by Nicholas Turner and Kisuk Lee.
