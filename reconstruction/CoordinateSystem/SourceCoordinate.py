"""
setup image coordinate system
"""
import math
import numpy as np
from Configuration import Config

class Source:

    def __init__(self, SAD, sourceAxisAngle):
        self.SAD = SAD
        self.sourceAxisAngle = sourceAxisAngle
        self.coordinate = np.zeros(3)

    def ComputeSourceCoordinate(self, SAD, sourceAxisAngle, axialAngle, iVew):
        stride = iVew * Config.deltaZ
        sourceCoord = np.zeros(3)
        sourceCoord[0] = SAD * math.sin(math.radians(sourceAxisAngle)) * math.sin(math.radians(axialAngle))
        sourceCoord[1] = -SAD * math.sin(math.radians(sourceAxisAngle)) * math.cos(math.radians(axialAngle))
        sourceCoord[2] = SAD * math.cos(math.radians(sourceAxisAngle)) + stride

        self.coordinate = np.round(sourceCoord, 8)


def SetupSourceCoordinate():

    source = Source(Config.SAD, Config.sourceAxisAngle)

    return source