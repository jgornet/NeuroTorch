# <img alt="NeuroTorch" src="/docs/images/NeuroTorch%20Logo.svg?raw=true" height="80">

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


## About

This project was created by James Gornet. Significant features were
rewritten from code provided by Nicholas Turner and Kisuk Lee.
