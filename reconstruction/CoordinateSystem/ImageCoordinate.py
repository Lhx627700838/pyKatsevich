"""
setup image coordinate system
"""
import numpy as np
from Configuration import Config

class Image:
    def __init__(self, dimension, resolution, offset, waterMu):
        self.dimension = dimension
        self.resolution = resolution
        self.offset = offset
        self.data = np.zeros([self.dimension[0], self.dimension[1], self.dimension[2]])
        self.waterMu = waterMu

    def ComputeImageVoxelCoordinate(self, dimension, resolution, offset):
        x = (np.arange(dimension[0]) - dimension[0] / 2 + 0.5) * resolution[0] + offset[0]
        y = (np.arange(dimension[1]) - dimension[1] / 2 + 0.5) * resolution[1] + offset[1]
        z = (np.arange(dimension[2]) - dimension[2] / 2 + 0.5) * resolution[2] + offset[2]
        
        # meshgrid indexing: ij 保持x,y,z的顺序
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # shape: (Nx, Ny, Nz)

        imageVoxelCoordinate = np.stack([X, Y, Z], axis=0)  # shape: (3, Nx, Ny, Nz)
        self.voxelCoordinate = np.round(imageVoxelCoordinate, 8)


def SetupSimulationImageCoordinate():

    image = Image(Config.simImageDimension, Config.simImageResolution, Config.simImageOffset, Config.waterMu)
    image.ComputeImageVoxelCoordinate(image.dimension, image.resolution, image.offset)

    return image


def SetupReconstructionImageCoordinate():

    image = Image(Config.reconImageDimension, Config.reconImageResolution, Config.reconImageOffset, Config.waterMu)
    image.ComputeImageVoxelCoordinate(image.dimension, image.resolution, image.offset)

    return image