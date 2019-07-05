import fnmatch
import os

import h5py
import numpy as np
import os.path
import tifffile as tif

from neurotorch.datasets.dataset import Array, PooledVolume, Volume
from neurotorch.datasets.datatypes import BoundingBox, Vector

from lxml import etree


class TiffVolume(Volume):
    def __init__(self, tiff_file, bounding_box: BoundingBox,
                 iteration_size: BoundingBox=BoundingBox(Vector(0, 0, 0),
                                                         Vector(128, 128, 32)),
                 stride: Vector=Vector(64, 64, 16)):
        """
        Loads a TIFF stack file or a directory of TIFF files and creates a
corresponding three-dimensional volume dataset
        :param tiff_file: Either a TIFF stack file or a directory
containing TIFF files
        :param chunk_size: Dimensions of the sample subvolume
        """
        # Set TIFF file and bounding box
        self.setFile(tiff_file)
        super().__init__(bounding_box, iteration_size, stride)

    def setFile(self, tiff_file):
        if os.path.isfile(tiff_file) or os.path.isdir(tiff_file):
            self.tiff_file = tiff_file
        else:
            raise IOError("{} was not found".format(tiff_file))

    def getFile(self):
        return self.tiff_file

    def get(self, bounding_box):
        return self.getArray().get(bounding_box)

    def __enter__(self):
        if os.path.isfile(self.getFile()):
            try:
                print("Opening {}".format(self.getFile()))
                array = tif.imread(self.getFile())

            except IOError:
                raise IOError("TIFF file {} could not be " +
                              "opened".format(self.getFile()))

        elif os.path.isdir(self.getFile()):
            tiff_list = os.listdir(self.getFile())
            tiff_list = filter(lambda f: fnmatch.fnmatch(f, '*.tif'),
                               tiff_list)

            if tiff_list:
                array = tif.TiffSequence(tiff_list).asarray()

        else:
            raise IOError("{} was not found".format(self.getFile()))

        array = Array(array, bounding_box=self.getBoundingBox(),
                      iteration_size=self.getIterationSize(),
                      stride=self.getStride())
        self.setArray(array)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.setArray(None)

    def _indexToBoundingBox(self, idx):
        return self.getArray()._indexToBoundingBox(idx)


class Hdf5Volume(Volume):
    def __init__(self, hdf5_file, dataset, bounding_box: BoundingBox,
                 iteration_size: BoundingBox=BoundingBox(Vector(0, 0, 0),
                                                         Vector(128, 128, 20)),
                 stride: Vector=Vector(64, 64, 10)):
        """
        Loads a HDF5 dataset and creates a corresponding three-dimensional
volume dataset

        :param hdf5_file: A HDF5 file path
        :param dataset: A HDF5 dataset name
        :param chunk_size: Dimensions of the sample subvolume
        """
        self.setFile(hdf5_file)
        self.setDataset(dataset)
        super().__init__(bounding_box, iteration_size, stride)

    def setFile(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def getFile(self):
        return self.hdf5_file

    def setDataset(self, hdf5_dataset):
        self.hdf5_dataset = hdf5_dataset

    def getDataset(self):
        return self.hdf5_dataset

    def __enter__(self):
        if os.path.isfile(self.getFile()):
            with h5py.File(self.getFile(), 'r') as f:
                array = f[self.getDataset()].value
                array = Array(array, bounding_box=self.getBoundingBox(),
                              iteration_size=self.getIterationSize(),
                              stride=self.getStride())
                self.setArray(array)

    def __exit__(self, exc_type, exc_value, traceback):
        self.setArray(None)


class BigDataVolume(PooledVolume):
    def __init__(self, xml_file, stack_size: int=5,
                 iteration_size: BoundingBox=BoundingBox(Vector(0, 0, 0),
                                                         Vector(128, 128, 20)),
                 stride: Vector=Vector(64, 64, 10)):
        """
        Loads a BigDataViewer dataset and creates a corresponding
        three-dimensional volume dataset

        :param xml_file: A BigDataViewer XML file path
        """
        self.setFile(xml_file)
        self._parse_xml(self.getFile())
        super().__init__(self.gatherVolumes(), stack_size,
                         iteration_size, stride)

    def _parse_xml(self, xml_file):
        with open(xml_file, 'r') as f:
            tree = etree.parse(f)

        # Gather volume indices
        index_list = tree.findall('.//ViewSetups/ViewSetup/id')
        index_list = ["s{}".format(index.text.zfill(2))
                      for index in index_list]

        # Gather volume sizes
        size_list = tree.findall('.//ViewSetups/ViewSetup/size')
        size_list = [tuple(map(int, size.text.split(' ')))
                     for size in size_list]

        # Create volume list
        volume_spec_list = []
        for index, size in zip(index_list, size_list):
            volume_spec_list.append({
                "path": "/t00000/{}/0/cells".format(index),
                "index": index,
                "size": size
            })
        self.volume_spec_list = volume_spec_list

        # Set dataset file
        dataset_file = tree.find('.//SequenceDescription/ImageLoader/hdf5').text
        if not os.path.isfile(dataset_file):
            raise OSError("{} is not a valid file".format(dataset_file))
        self.dataset_file = dataset_file

    def gatherVolumes(self):
        volume_list = []
        translation_shift = 0
        for index, volume_spec in enumerate(self.volume_spec_list):
            volume_list.append(Hdf5Volume(self.dataset_file,
                                          volume_spec['path'],
                                          BoundingBox(*volume_spec['size']) + \
                                          Vector(translation_shift, 0, 0)))
            translation_shift += volume_spec['size'][0]

        return volume_list

    def setFile(self, xml_file):
        if not os.path.isfile(xml_file):
            raise OSError("{} is not a valid file".format(xml_file))

        self.xml_file = xml_file

    def getFile(self):
        return self.xml_file
