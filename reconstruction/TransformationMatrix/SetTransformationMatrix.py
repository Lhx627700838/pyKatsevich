"""
defines common transformation matrices
"""
import math
import numpy as np

def DetectorTiltTransformation(detectorTiltAngle):

    detectorTiltTransformationMatrix = np.zeros([4, 4])

    detectorTiltTransformationMatrix[0, 0] = 1
    detectorTiltTransformationMatrix[1, 1] = math.cos(math.radians(detectorTiltAngle))
    detectorTiltTransformationMatrix[1, 2] = -math.sin(math.radians(detectorTiltAngle))
    detectorTiltTransformationMatrix[2, 1] = math.sin(math.radians(detectorTiltAngle))
    detectorTiltTransformationMatrix[2, 2] = math.cos(math.radians(detectorTiltAngle))
    detectorTiltTransformationMatrix[3, 3] = 1

    return detectorTiltTransformationMatrix


def AxialAngleTransformation(axialAngle):

    axialAngleTransformationMatrix = np.zeros([4, 4])
    axialAngleTransformationMatrix[0, 0] = math.cos(math.radians(axialAngle))
    axialAngleTransformationMatrix[1, 0] = math.sin(math.radians(axialAngle))
    axialAngleTransformationMatrix[0, 1] = -math.sin(math.radians(axialAngle))
    axialAngleTransformationMatrix[1, 1] = math.cos(math.radians(axialAngle))
    axialAngleTransformationMatrix[2, 2] = 1
    axialAngleTransformationMatrix[3, 3] = 1

    return axialAngleTransformationMatrix


def SourceAxisTiltTransformation(sourceAxisAngle):

    sourceAxisTiltTransformationMatrix = np.zeros([4, 4])

    sourceAxisTiltTransformationMatrix[0, 0] = 1
    sourceAxisTiltTransformationMatrix[1, 1] = math.cos(math.radians(sourceAxisAngle))
    sourceAxisTiltTransformationMatrix[1, 2] = -math.sin(math.radians(sourceAxisAngle))
    sourceAxisTiltTransformationMatrix[2, 1] = math.sin(math.radians(sourceAxisAngle))
    sourceAxisTiltTransformationMatrix[2, 2] = math.cos(math.radians(sourceAxisAngle))
    sourceAxisTiltTransformationMatrix[3, 3] = 1

    return sourceAxisTiltTransformationMatrix