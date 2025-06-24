"""
MLEM Reconstruction
"""
import copy
import numpy as np
from Configuration import Config
from Utils.SaveDICOM import SaveDICOM
from Projection.BackProjection import BackProjection
from Projection.ForwardProjection import ForwardProjection


def MLEM(projectionData, image, source, detector):

    reconImage = copy.deepcopy(image)
    reconImageData = np.ones([image.dimension[0], image.dimension[1], image.dimension[2]])
    reconImage.data = reconImageData

    if Config.evenlyDistributedView:
        nView = Config.nView
    else:
        nView = len(Config.viewSet)

    if nView != np.shape(projectionData)[2]:
        raise Exception("nView in Config.py different from projection data")

    print('Generating normalization image')
    normImage = np.zeros([image.dimension[0], image.dimension[1], image.dimension[2]])
    normImage = normImage + BackProjection(np.ones([detector.dimension[0], detector.dimension[1], nView]), image, source, detector)

    # iteration
    for iter in range(Config.mlemIteration):

        print('MLEM iteration ' + str(iter + 1) + ' of ' + str(Config.mlemIteration), end='\r')

        projRatio = projectionData / ForwardProjection(reconImage, source, detector)
        projRatio[np.isnan(projRatio)] = 0
        projRatio[np.isinf(projRatio)] = 0

        imageRatio = BackProjection(projRatio, image, source, detector) / normImage

        imageRatio[np.isnan(imageRatio)] = 0
        imageRatio[np.isinf(imageRatio)] = 0

        reconImageData = reconImageData * imageRatio
        reconImageData[reconImageData < 0] = 0

        reconImage.data = reconImageData

        SaveDICOM(reconImage, 'outputMLEMIntermediate/outputReconImageIteration' + str(iter+1))

    return reconImage