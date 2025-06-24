"""
set up scanner and imaging coordinate system
"""
from CoordinateSystem.SourceCoordinate import SetupSourceCoordinate
from CoordinateSystem.DetectorCoordinate import SetupDetectorCoordinate

def SetupCoordinate():

    source = SetupSourceCoordinate()
    detector = SetupDetectorCoordinate()

    return source, detector
