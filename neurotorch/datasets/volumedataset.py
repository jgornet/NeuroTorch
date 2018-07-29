from torch.utils.data import Dataset as _Dataset
import numpy as np
from abc import (ABC, abstractmethod)
from neurotorch.datasets.datatypes import BoundingBox, Vector
import fnmatch
import tifffile as tif
import os
import os.path
import h5py


class Data:
    """
    An encapsulating object for communicating volume data
    """
    def __init__(self, array, bounding_box):
        self._setBoundingBox(bounding_box)
        self._setArray(array)

    def getBoundingBox(self):
        return self.bounding_box

    def _setBoundingBox(self, bounding_box):
        if not isinstance(bounding_box, BoundingBox):
            raise ValueError("bounding_box must have type BoundingBox")

        self.bounding_box = bounding_box

    def getArray(self):
        return self.array

    def _setArray(self, array):
        self.array = array

    def getSize(self):
        return self.getBoundingBox().getSize()

    def getDimension(self):
        return self.getBoundingBox().getDimension()


class Dataset(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get(self, *args):
        pass

    @abstractmethod
    def set(self, *args):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self):
            result = self.__getitem__(self.index)
            self.index += 1
            return result
        else:
            self.index = 0
            raise StopIteration


class Volume(Dataset):
    """
    A dataset containing a 3D Numpy array
    """
    def __init__(self, array, iteration_size=BoundingBox(Vector(0, 0, 0),
                                                         Vector(128, 128, 20)),
                 stride=Vector(64, 64, 10)):
        if isinstance(array, np.ndarray):
            self._setArray(array)
        elif isinstance(array, BoundingBox):
            self.createArray(array)
        else:
            raise ValueError("array must be an ndarray or a BoundingBox")

        self.setIteration(iteration_size=iteration_size,
                          stride=stride)
        super().__init__()

    def get(self, bounding_box):
        if bounding_box.isDisjoint(self.getBoundingBox()):
            raise ValueError("Bounding box must be inside dataset " +
                             "dimensions instead bounding box is {} while the dataset dimensions are {}".format(bounding_box, self.getBoundingBox()))

        sub_bounding_box = bounding_box.intersect(self.getBoundingBox())
        array = self.getArray(sub_bounding_box)

        before_pad = bounding_box.getEdges()[0] - sub_bounding_box.getEdges()[0]
        after_pad = sub_bounding_box.getEdges()[1] - bounding_box.getEdges()[1]

        if before_pad != Vector(0, 0, 0) and after_pad != Vector(0, 0, 0):
            pad_size = (before_pad.getComponents(),
                        after_pad.getComponents())
            array = np.pad(array, pad_size=pad_size, mode="constant")

        return Data(array, bounding_box)

    def set(self, data):
        bounding_box = data.getBoundingBox()
        array = data.getArray()

        if not bounding_box.isSubset(self.getBoundingBox()):
            raise ValueError("The bounding box must be a subset of the "
                             " volume")

        edge1, edge2 = bounding_box.getEdges()
        x1, y1, z1 = edge1.getComponents()
        x2, y2, z2 = edge2.getComponents()

        self.array[z1:z2, y1:y2, x1:x2] = array

    def createArray(self, bounding_box):
        size = bounding_box.getSize().getComponents()[::-1]
        volume = np.zeros(size)
        self._setArray(volume)

    def getArray(self, bounding_box=None):
        if bounding_box is None:
            return self.array

        else:
            if not bounding_box.isSubset(self.getBoundingBox()):
                raise ValueError("The bounding box must be a subset" +
                                 " of the volume")

            edge1, edge2 = bounding_box.getEdges()
            x1, y1, z1 = edge1.getComponents()
            x2, y2, z2 = edge2.getComponents()

            return self.array[z1:z2, y1:y2, x1:x2]

    def _setArray(self, array):
        self.array = array

    def getBoundingBox(self):
        return BoundingBox(Vector(0, 0, 0),
                           Vector(*self.getArray().shape[::-1]))

    def setIteration(self, iteration_size: BoundingBox, stride: Vector):
        if not isinstance(iteration_size, BoundingBox):
            raise ValueError("iteration_size must have type BoundingBox"
                             + " instead it has type {}".format(type(iteration_size)))

        if not isinstance(stride, Vector):
            raise ValueError("stride must have type Vector")

        if not iteration_size.isSubset(self.getBoundingBox()):
            raise ValueError("iteration_size must be smaller than volume size")

        self.setIterationSize(iteration_size)
        self.setStride(stride)

        def ceil(x):
            return int(round(x))

        self.element_vec = Vector(*map(lambda L, l, s: ceil((L-l)/s+1),
                                       self.getBoundingBox().getEdges()[1].getComponents(),
                                       self.iteration_size.getEdges()[1].getComponents(),
                                       self.stride.getComponents()))

        self.index = 0

    def setIterationSize(self, iteration_size):
        self.iteration_size = BoundingBox(Vector(0, 0, 0),
                                          iteration_size.getSize())

    def setStride(self, stride):
        self.stride = stride

    def getIterationSize(self):
        return self.iteration_size

    def getStride(self):
        return self.stride

    def __len__(self):
        return self.element_vec[0]*self.element_vec[1]*self.element_vec[2]

    def __getitem__(self, idx):
        if idx >= len(self):
            self.index = 0
            raise StopIteration

        element_vec = np.unravel_index(idx,
                                       dims=self.element_vec.getComponents())

        element_vec = Vector(*element_vec)
        bounding_box = self.iteration_size+self.stride*element_vec
        result = self.get(bounding_box)

        return result


class TiffVolume(Volume):
    def __init__(self, tiff_file, *args, **kwargs):
        """
        Loads a TIFF stack file or a directory of TIFF files and creates a
corresponding three-dimensional volume dataset
        :param tiff_file: Either a TIFF stack file or a directory containing TIFF files
        :param chunk_size: Dimensions of the sample subvolume
        """
        if os.path.isfile(tiff_file):
            try:
                array = tif.imread(tiff_file)

            except IOError:
                raise IOError("TIFF file {} could not be " +
                              "opened".format(tiff_file))

        elif os.path.isdir(tiff_file):
            tiff_list = os.listdir(tiff_file)
            tiff_list = filter(lambda f: fnmatch.fnmatch(f, '*.tif'),
                               tiff_list)

            if tiff_list:
                array = tif.TiffSequence(tiff_list).asarray()

        else:
            raise IOError("{} was not found".format(tiff_file))

        super().__init__(array, *args, **kwargs)


class Hdf5Volume(Volume):
    def __init__(self, hdf5_file, dataset):
        """
        Loads a HDF5 dataset and creates a corresponding three-dimensional volume dataset
        :param hdf5_file: A HDF5 file path
        :param dataset: A HDF5 dataset name
        :param chunk_size: Dimensions of the sample subvolume
        """
        self.hdf5_file = h5py.File(hdf5_file)
        array = self.hdf5_file[dataset].value

        super(array)


class TorchVolume(_Dataset):
    def __init__(self, volume):
        self.setVolume(volume)
        super().__init__()

    def __len__(self):
        return len(self.getVolume())

    def __getitem__(self, idx):
        if isinstance(self.getVolume(), AlignedVolume):
            data_list = [self.toTorch(data) for data in self.getVolume()[idx]]
            return data_list
        else:
            return self.getVolume()[idx].getArray()

    def toTorch(self, data):
        torch_data = data.getArray().astype(np.float)
        torch_data = torch_data.reshape(1, *torch_data.shape)
        return torch_data

    def setVolume(self, volume):
        self.volume = volume

    def getVolume(self):
        return self.volume


class AlignedVolume(Volume):
    def __init__(self, volumes, iteration_size=None, stride=None):
        if iteration_size is None:
            iteration_size = volumes[0].getIterationSize()
        if stride is None:
            stride = volumes[0].getStride()
        self.setVolumes(volumes)
        self.setIteration(iteration_size, stride)

    def getBoundingBox(self):
        return self.getVolumes()[0].getBoundingBox()

    def setVolumes(self, volumes):
        self.volumes = volumes

    def addVolume(self, volume):
        self.volumes.append(volume)

    def getVolumes(self):
        return self.volumes

    def setIteration(self, iteration_size, stride):
        for volume in self.getVolumes():
            volume.setIteration(iteration_size, stride)

    def get(self, bounding_box):
        result = [volume.get(bounding_box)
                  for volume in self.getVolumes()]
        return result

    def set(self, array, bounding_box):
        pass

    def __len__(self):
        return len(self.getVolumes()[0])

    def __getitem__(self, idx):
        result = [volume[idx] for volume in self.getVolumes()]
        return result
