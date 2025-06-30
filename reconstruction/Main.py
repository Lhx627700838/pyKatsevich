"""
top level main function
"""
import numpy as np
import subprocess
import time
import tifffile
from CoordinateSystem.SetupCoordinate import SetupCoordinate
from Reconstruction.ImageReconstruction import ImageReconstruction
from CoordinateSystem.ImageCoordinate import SetupSimulationImageCoordinate, SetupReconstructionImageCoordinate
from Configuration import Config
from PIL import Image
import os
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv

if __name__ == "__main__":

    tStart = time.time()

    # set up the scanner coordinate system
    source, detector = SetupCoordinate()
    # First compile the c-projector
    # subprocess.call(["gcc", "-shared", "-Wl,-soname,BackProject", "-o", "Projection/BackProject.so", \
    #                     "-fPIC", "Projection/BackProject.c"])
    # set up simImage coordinate system

    projectionData = tifffile.imread('filtered_proj6.tif')
    projectionData = np.transpose(projectionData,[2,1,0])
    print(projectionData.shape)
    

    reconImage = ImageReconstruction(projectionData, source, detector)
    recondata = reconImage.data
    recondata = recondata.astype(np.float32)
    recondata = np.transpose(recondata,[2,1,0])
    print(type(recondata))
    print(type(reconImage.data))
    tifffile.imwrite('recon_voxel.tif', recondata)
    

    tEnd = time.time()
    #print('Sim Processing time: ' + str(np.round(tEndsim-tStartsim, 1)) + ' sec.')
    print('Total Processing time: ' + str(np.round(tEnd-tStart, 1)) + ' sec.')