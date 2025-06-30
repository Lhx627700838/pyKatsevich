
"""
back projection
"""
from Configuration import Config
import numpy as np
import sys
import multiprocessing
import math
import ctypes

def CalcNormalVector(point1, point2, point3):

    normalVector = [(point2[1] - point1[1]) * (point3[2] - point1[2]) - (point3[1] - point1[1]) * (point2[2] - point1[2]),
                    (point2[2] - point1[2]) * (point3[0] - point1[0]) - (point3[2] - point1[2]) * (point2[0] - point1[0]),
                    (point2[0] - point1[0]) * (point3[1] - point1[1]) - (point3[0] - point1[0]) * (point2[1] - point1[1])]

    return normalVector


def CalcLinePlaneIntersection(normalVector, pointOnPlane, firstPointOnLine, seconPointOnLine):

    u = firstPointOnLine - pointOnPlane
    v = seconPointOnLine - firstPointOnLine
    N = -np.dot(normalVector, u)
    D = np.dot(normalVector, v)
    sI = N / D
    intersectionCoordinate = firstPointOnLine + sI * v

    return intersectionCoordinate


def CalcInterpolatedDetectorReading(pointCoordinate, projectionData, detector):

    xUnitVector = (detector.coordinate[0:3, 1, 0] - detector.coordinate[0:3, 0, 0]) / detector.resolution[0]
    yUnitVector = (detector.coordinate[0:3, 0, 1] - detector.coordinate[0:3, 0, 0]) / detector.resolution[1]

    # coordinate index of intersection point on detector
    pointXCoordinate = np.dot((pointCoordinate - detector.coordinate[0:3, 0, 0]), xUnitVector) / detector.resolution[0]
    pointYCoordinate = np.dot((pointCoordinate - detector.coordinate[0:3, 0, 0]), yUnitVector) / detector.resolution[1]

    # return nan if point is outside of detector panel
    if (pointXCoordinate < -0.5) or (pointYCoordinate < -0.5) or (pointXCoordinate > detector.dimension[0] - 0.5) or (pointYCoordinate > detector.dimension[1] - 0.5):
        return np.nan

    # handle cases where intersection is on the edge detectors and on detector pixel centers
    if (pointXCoordinate <= 0):
        xCoordFirst = 0
        xCoordSecond = 0
        xDistanceFirst = 1
        xDistanceSecond = 1
    elif (pointXCoordinate >= detector.dimension[0] - 1):
        xCoordFirst = detector.dimension[0] - 1
        xCoordSecond = detector.dimension[0] - 1
        xDistanceFirst = 1
        xDistanceSecond = 1
    elif (pointXCoordinate == np.round(pointXCoordinate)):
        xCoordFirst = pointXCoordinate
        xCoordSecond = pointXCoordinate
        xDistanceFirst = 1
        xDistanceSecond = 1
    else:
        xCoordFirst = np.floor(pointXCoordinate)
        xCoordSecond = np.ceil(pointXCoordinate)
        xDistanceFirst = pointXCoordinate - xCoordFirst
        xDistanceSecond = xCoordSecond - pointXCoordinate

    if (pointYCoordinate <= 0):
        yCoordFirst = 0
        yCoordSecond = 0
        yDistanceFirst = 1
        yDistanceSecond = 1
    elif (pointYCoordinate >= detector.dimension[1] - 1):
        yCoordFirst = detector.dimension[1] - 1
        yCoordSecond = detector.dimension[1] - 1
        yDistanceFirst = 1
        yDistanceSecond = 1
    elif (pointYCoordinate == np.round(pointYCoordinate)):
        yCoordFirst = pointYCoordinate
        yCoordSecond = pointYCoordinate
        yDistanceFirst = 1
        yDistanceSecond = 1
    else:
        yCoordFirst = np.floor(pointYCoordinate)
        yCoordSecond = np.ceil(pointYCoordinate)
        yDistanceFirst = pointYCoordinate - yCoordFirst
        yDistanceSecond = yCoordSecond - pointYCoordinate

    distanceSum = (xDistanceFirst + xDistanceSecond + yDistanceFirst + yDistanceSecond) * 2

    xCoordFirst = int(xCoordFirst)
    xCoordSecond = int(xCoordSecond)
    yCoordFirst = int(yCoordFirst)
    yCoordSecond = int(yCoordSecond)

    # compute interpolated detector reading
    interpolatedDetectorReading = (projectionData[xCoordFirst, yCoordFirst] * (xDistanceFirst + yDistanceFirst) + \
                                  projectionData[xCoordFirst, yCoordSecond] * (xDistanceFirst + yDistanceSecond) + \
                                  projectionData[xCoordSecond, yCoordFirst] * (xDistanceSecond + yDistanceFirst) + \
                                  projectionData[xCoordSecond, yCoordSecond] * (xDistanceSecond + yDistanceSecond)) / distanceSum

    if np.isnan(interpolatedDetectorReading):
        raise ValueError("NaN value encountered in projection data interpolation")

    return interpolatedDetectorReading


def BackProjectView(iView, viewSet, projectionView, image, source, detector):

    source.ComputeSourceCoordinate(source.SAD, source.sourceAxisAngle, viewSet[iView])
    sourceCoordinate = source.coordinate
    detector.ComputeDetectCoordinate(detector.SAD, detector.SDD, detector.dimension, detector.resolution,
                                     detector.offset, detector.detectorAxisAngle, detector.detectorTiltAngle,
                                     viewSet[iView])
    dbeta = 2 * math.pi / Config.nView  # needs to math with the scan rotation (clockwise or counter clockwise)
    beta = iView * dbeta
    betaVector = [math.sin(beta), math.cos(beta)]
    sourceSAD = source.SAD

    normalVectorToDetectorPlane = CalcNormalVector(detector.coordinate[0:3, 0, 0], detector.coordinate[0:3, 0, 1],
                                                   detector.coordinate[0:3, 1, 0])

    if Config.projectorPlatform == 'c':
        backProjectHandle = ctypes.CDLL('./Projection/BackProject.so')
        backProjectHandle.BackProjectFunc.argtypes = (ctypes.POINTER(ctypes.c_int),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.c_float,
                                                            ctypes.POINTER(ctypes.c_int),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.c_float,
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float))

        detectorCoordinate = detector.coordinate
        detectorCoordinate = np.ndarray.flatten(detectorCoordinate)

        imageCoordinate = image.voxelCoordinate
        imageCoordinate = np.ndarray.flatten(imageCoordinate)

        projectionView1D = np.ndarray.flatten(projectionView)

        xUnitVector = (detector.coordinate[0:3, 1, 0] - detector.coordinate[0:3, 0, 0]) / detector.resolution[0]
        yUnitVector = (detector.coordinate[0:3, 0, 1] - detector.coordinate[0:3, 0, 0]) / detector.resolution[1]

        arrayTypeImageDimension = ctypes.c_int * 3
        arrayTypeImageResolution = ctypes.c_float * 3
        arrayTypeImageCoordinate = ctypes.c_float * len(imageCoordinate)
        arrayTypeSourceCoordinate = ctypes.c_float * 3
        arrayTypeDetectorDimension = ctypes.c_int * len(detector.dimension)
        arrayTypeDetectorResolution = ctypes.c_float * len(detector.resolution)
        arrayTypeDetectorCoordinate = ctypes.c_float * len(detectorCoordinate)
        arrayTypeXUnitVector = ctypes.c_float * 3
        arrayTypeYUnitVector = ctypes.c_float * 3
        arrayTypeNormalVectorToDetectorPlane = ctypes.c_float * 3
        arrayTypeProjectionView = ctypes.c_float * len(projectionView1D)
        arrayTypeBetaVector = ctypes.c_float * 2

        backProjectedDataInit = list(np.zeros(image.dimension[0] * image.dimension[1] * image.dimension[2]).astype(float))  # Create List with underlying memory
        backProjectedDataPointer = (ctypes.c_float * len(backProjectedDataInit))(*backProjectedDataInit)  # Create ctypes pointer to underlying memory

        backProjectHandle.BackProjectFunc(arrayTypeImageDimension(*image.dimension),
                                                arrayTypeImageResolution(*image.resolution),
                                                arrayTypeImageCoordinate(*imageCoordinate),
                                                arrayTypeSourceCoordinate(*source.coordinate),
                                                ctypes.c_float(sourceSAD),
                                                arrayTypeDetectorDimension(*detector.dimension),
                                                arrayTypeDetectorCoordinate(*detectorCoordinate),
                                                arrayTypeDetectorResolution(*detector.resolution),
                                                arrayTypeNormalVectorToDetectorPlane(*normalVectorToDetectorPlane),
                                                arrayTypeXUnitVector(*xUnitVector),
                                                arrayTypeYUnitVector(*yUnitVector),
                                                arrayTypeBetaVector(*betaVector),
                                                ctypes.c_float(dbeta),
                                                arrayTypeProjectionView(*projectionView1D),
                                                backProjectedDataPointer)

        backProjectedData = np.array(backProjectedDataPointer[:]).reshape((image.dimension[0], image.dimension[1], image.dimension[2]))

    if Config.projectorPlatform == 'python':

        backProjectedData = np.zeros([image.dimension[0], image.dimension[1], image.dimension[2]])

        for iX in range(image.dimension[0]):
            for iY in range(image.dimension[1]):
                for iZ in range(image.dimension[2]):
                    imageCoordinate = image.voxelCoordinate[0:3, iX, iY, iZ]
                    intersectionCoordinate = CalcLinePlaneIntersection(normalVectorToDetectorPlane,
                                                                       detector.coordinate[0:3, 0, 0],
                                                                       sourceCoordinate, imageCoordinate)
                    magni = sourceSAD / (sourceSAD + imageCoordinate[1] * betaVector[1] - imageCoordinate[0] * betaVector[0])  # needs double check later
                    backProjectedData[iX, iY, iZ] = magni**2 * CalcInterpolatedDetectorReading(intersectionCoordinate, projectionView, detector) * dbeta

    return (iView, backProjectedData)


def GetResult(result):
    global results
    results.append(result)


def BackProjection(projectionData, image, source, detector):

    reconImageData = np.zeros([image.dimension[0], image.dimension[1], image.dimension[2]])

    if Config.evenlyDistributedView:
        nView = Config.nView
        viewSet = np.arange(nView) * (360 / nView)
    else:
        nView = len(Config.viewSet)
        viewSet = Config.viewSet

    if nView != np.shape(projectionData)[2]:
        sys.exit("nView in Config.py different from projection data")

    global results
    results = []

    if Config.multiprocessing:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        for iView in range(nView):
            pool.apply_async(BackProjectView, args=(iView, viewSet, projectionData[:, :, iView], image, source, detector), callback=GetResult)
        pool.close()
        pool.join()

        for iView in range(nView):
            reconImageData = reconImageData + np.array(results[iView][1])
    else:
        for iView in range(nView):
            GetResult(BackProjectView(iView, viewSet, projectionData[:, :, iView], image, source, detector))

        for iView in range(nView):
            reconImageData = reconImageData + np.array(results[iView][1])

    return reconImageData