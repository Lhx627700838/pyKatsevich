"""
image reconstruction from projection data
"""
from Configuration import Config
from Reconstruction.FilteredBackprojection import FilteredBackprojection
from Reconstruction.MLEM import MLEM
from CoordinateSystem.ImageCoordinate import SetupReconstructionImageCoordinate

def ImageReconstruction(projectionData, source, detector):

    print("Reconstructing image using " + Config.reconMethod)
    image = SetupReconstructionImageCoordinate()
    print("X范围:", image.voxelCoordinate[0].min(), image.voxelCoordinate[0].max())
    print("Y范围:", image.voxelCoordinate[1].min(), image.voxelCoordinate[1].max())
    print("Z范围:", image.voxelCoordinate[2].min(), image.voxelCoordinate[2].max())
    if Config.reconMethod == 'fbp':
        reconImage = FilteredBackprojection(projectionData, image, source, detector)

    if Config.reconMethod == 'mlem':
        reconImage = MLEM(projectionData, image, source, detector)

    return reconImage
