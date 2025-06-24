"""
back projection
"""
from Configuration import Config
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import numpy as np
import sys
import multiprocessing
import math
import ctypes
import time
import Utils.Visualization.VisualizeSystem as VisualizeSystem
import matplotlib.pyplot as plt

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


def BackProjectView(detector_center_traj,source_trajectory, fig,ax,iView, viewSet, projectionView, image, source, detector, cuda_id):
    
    source.ComputeSourceCoordinate(source.SAD, source.sourceAxisAngle, viewSet[iView],iView)
    sourceCoordinate = source.coordinate
    detector.ComputeDetectCoordinate(detector.SAD, detector.SDD, detector.dimension, detector.resolution,
                                     detector.offset, detector.detectorAxisAngle, detector.detectorTiltAngle,
                                     viewSet[iView], iView, cuda_id)
    print(f"Starting FBP for view {iView} on CUDA device {cuda_id}")

        
    source_trajectory.append(source.coordinate.reshape(3))
    detector_center = np.mean(detector.coordinate.reshape(3, -1), axis=1)
    detector_center_traj.append(detector_center)
    VisualizeSystem.visualize_system_dynamic(fig, ax, image, source, detector,source_trajectory,detector_center_traj)

    dbeta = 2 * math.pi / Config.nView  # needs to math with the scan rotation (clockwise or counter clockwise)
    beta = iView * dbeta
    betaVector = [math.sin(beta), math.cos(beta)]
    sourceSAD = source.SAD
    
    normalVectorToDetectorPlane = CalcNormalVector(detector.coordinate[0:3, 0, 0], detector.coordinate[0:3, 0, 1],
                                                   detector.coordinate[0:3, 1, 0])

    if Config.projectorPlatform in ['c']:
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
        
    if Config.projectorPlatform == 'cuda':
        tStartcuda = time.time()
        try:
            #print("FBP CUDA initialized.")
            device = drv.Device(cuda_id)  # 定义 device 变量
            context = device.make_context()
            #print(f"FBP Running on device: {cuda_id}, {device.name()}")
        except Exception as e:
            print(f"Error initializing CUDA: {e}")
    # CUDA 核函数
        cuda_code = """
        __device__ float* CalcLinePlaneIntersection(float normalVector[], float pointOnPlane[], float firstPointOnLine[], float seconPointOnLine[]) {
            float u[3], v[3];
            float sI;
            float N = 0, D = 0;
            static float intersectionCoordinate[3];

            for (int i = 0; i < 3; i++) {
                u[i] = firstPointOnLine[i] - pointOnPlane[i];
                v[i] = seconPointOnLine[i] - firstPointOnLine[i];
                N = N - normalVector[i] * u[i];
                D = D + normalVector[i] * v[i];
            }
            sI = N / D;

            for (int i = 0; i < 3; i++) {
                intersectionCoordinate[i] = firstPointOnLine[i] + sI * v[i];
            }

            return intersectionCoordinate;
        }

        __device__ float CalcInterpolatedDetectorReading(float pointCoordinate[], float projectionData[], float xUnitVector[], float yUnitVector[], float detectorResolution[], int detectorDimension[], float detectorCoordinateFirst[]) {
            float pointXCoordinate = 0.0, pointYCoordinate = 0.0;
            int xCoordFirst, xCoordSecond, yCoordFirst, yCoordSecond;
            float xDistanceFirst, xDistanceSecond, yDistanceFirst, yDistanceSecond;
            float interpolatedDetectorReading;

            for (int i = 0; i < 3; i++) {
                pointXCoordinate += (pointCoordinate[i] - detectorCoordinateFirst[i]) * xUnitVector[i];
                pointYCoordinate += (pointCoordinate[i] - detectorCoordinateFirst[i]) * yUnitVector[i];
            }
            pointXCoordinate /= detectorResolution[0];
            pointYCoordinate /= detectorResolution[1];
            //printf("pointXCoordinate: %f, pointYCoordinate: %f     ", pointXCoordinate,pointYCoordinate);


            if ((pointXCoordinate < -0.5) || (pointYCoordinate < -0.5) || (pointXCoordinate > (detectorDimension[0] - 0.5)) || (pointYCoordinate > (detectorDimension[1] - 0.5))) {
                return 0;
                printf("out of range");
            }

            if (pointXCoordinate <= 0) {
                xCoordFirst = 0;
                xCoordSecond = 0;
                xDistanceFirst = 1;
                xDistanceSecond = 1;
            } else if (pointXCoordinate >= (detectorDimension[0] - 1)) {
                xCoordFirst = detectorDimension[0] - 1;
                xCoordSecond = detectorDimension[0] - 1;
                xDistanceFirst = 1;
                xDistanceSecond = 1;
            } else if (pointXCoordinate == round(pointXCoordinate)) {
                xCoordFirst = pointXCoordinate;
                xCoordSecond = pointXCoordinate;
                xDistanceFirst = 1;
                xDistanceSecond = 1;
            } else {
                xCoordFirst = floor(pointXCoordinate);
                xCoordSecond = ceil(pointXCoordinate);
                xDistanceFirst = pointXCoordinate - xCoordFirst;
                xDistanceSecond = xCoordSecond - pointXCoordinate;
            }

            if (pointYCoordinate <= 0) {
                yCoordFirst = 0;
                yCoordSecond = 0;
                yDistanceFirst = 1;
                yDistanceSecond = 1;
            } else if (pointYCoordinate >= (detectorDimension[1] - 1)) {
                yCoordFirst = detectorDimension[1] - 1;
                yCoordSecond = detectorDimension[1] - 1;
                yDistanceFirst = 1;
                yDistanceSecond = 1;
            } else if (pointYCoordinate == round(pointYCoordinate)) {
                yCoordFirst = pointYCoordinate;
                yCoordSecond = pointYCoordinate;
                yDistanceFirst = 1;
                yDistanceSecond = 1;
            } else {
                yCoordFirst = floor(pointYCoordinate);
                yCoordSecond = ceil(pointYCoordinate);
                yDistanceFirst = pointYCoordinate - yCoordFirst;
                yDistanceSecond = yCoordSecond - pointYCoordinate;
            }

            float distanceSum = (xDistanceFirst + xDistanceSecond + yDistanceFirst + yDistanceSecond) * 2;

            interpolatedDetectorReading = (projectionData[xCoordFirst * detectorDimension[1] + yCoordFirst] * (xDistanceFirst + yDistanceFirst) +
                                        projectionData[xCoordFirst * detectorDimension[1] + yCoordSecond] * (xDistanceFirst + yDistanceSecond) +
                                        projectionData[xCoordSecond * detectorDimension[1] + yCoordFirst] * (xDistanceSecond + yDistanceFirst) +
                                        projectionData[xCoordSecond * detectorDimension[1] + yCoordSecond] * (xDistanceSecond + yDistanceSecond)) / distanceSum;

            return interpolatedDetectorReading;
        }

        __global__ void BackProjectFunc(int *imageDimension, float *imageResolution, float *imageCoordinate,
                                        float *sourceCoordinate, float sourceSAD,
                                        int *detectorDimension, float *detectorCoordinate, float *detectorResolution,
                                        float *normalVectorToDetectorPlane, float *xUnitVector, float *yUnitVector,
                                        float *betaVector, float dbeta,
                                        float *projectionView,
                                        float *backProjectedData) {
            int iX = blockIdx.x * blockDim.x + threadIdx.x;
            int iY = blockIdx.y * blockDim.y + threadIdx.y;
            int iZ = blockIdx.z * blockDim.z + threadIdx.z;

            if (iX >= imageDimension[0] || iY >= imageDimension[1] || iZ >= imageDimension[2]) return;

            float detectorCoordinateFirst[3];
            float imageCoordinateThis[3];
            float magni;
            float v;
            float *intersectionCoordinate;
            float interpolatedDetectorReading;

            int detectorIndexFirstX = 0;
            int detectorIndexFirstY = detectorDimension[0] * detectorDimension[1] * 1;
            int detectorIndexFirstZ = detectorDimension[0] * detectorDimension[1] * 2;

            detectorCoordinateFirst[0] = detectorCoordinate[detectorIndexFirstX];
            detectorCoordinateFirst[1] = detectorCoordinate[detectorIndexFirstY];
            detectorCoordinateFirst[2] = detectorCoordinate[detectorIndexFirstZ];

            for (int iCoord = 0; iCoord < 3; iCoord++) {
                imageCoordinateThis[iCoord] = imageCoordinate[iZ + imageDimension[2] * (iY + imageDimension[1] * (iX + imageDimension[0] * iCoord))];
            }

            intersectionCoordinate = CalcLinePlaneIntersection(normalVectorToDetectorPlane, detectorCoordinateFirst, sourceCoordinate, imageCoordinateThis);
            //printf("x:%f, y:%f, z:%f",intersectionCoordinate[0], intersectionCoordinate[1], intersectionCoordinate[2]);
            magni = sourceSAD / (sourceSAD + imageCoordinateThis[1] * betaVector[1] - imageCoordinateThis[0] * betaVector[0]);
            interpolatedDetectorReading = CalcInterpolatedDetectorReading(intersectionCoordinate, projectionView, xUnitVector, yUnitVector, detectorResolution, detectorDimension, detectorCoordinateFirst);
            //backProjectedData[(iX * imageDimension[1] + iY) * imageDimension[2] + iZ] = magni* magni * interpolatedDetectorReading * dbeta;
            v = sourceSAD - imageCoordinateThis[0]*betaVector[1]-imageCoordinateThis[1]*betaVector[0];
            backProjectedData[(iX * imageDimension[1] + iY) * imageDimension[2] + iZ] = interpolatedDetectorReading/v;
            //if (interpolatedDetectorReading!=0) printf("sourceSAD: %f, interpolatedDetectorReading: %f, backProjectedData: %f       ",sourceSAD, interpolatedDetectorReading , backProjectedData[(iX * imageDimension[1] + iY) * imageDimension[2] + iZ]);
        }
        """

        mod = SourceModule(cuda_code, options=["-allow-unsupported-compiler"])
        backproject_func = mod.get_function("BackProjectFunc")

        # 将图像、探测器、源等信息转换为一维数组，准备传递给 CUDA
        detectorCoordinate = np.ndarray.flatten(detector.coordinate)
        imageCoordinate = np.ndarray.flatten(image.voxelCoordinate)
        projectionView1D = np.ndarray.flatten(projectionView).astype(np.float32)
        imageDimension = np.array(image.dimension, dtype=np.int32)
        imageResolution = np.array(image.resolution, dtype=np.float32)
        imageCoordinate = np.ndarray.flatten(image.voxelCoordinate).astype(np.float32)
        sourceCoordinate = np.array(source.coordinate, dtype=np.float32)
        detectorCoordinate = np.ndarray.flatten(detector.coordinate).astype(np.float32)
        detectorDimension = np.array(detector.dimension, dtype=np.int32)
        detectorResolution = np.array(detector.resolution, dtype=np.float32)
        
        xUnitVector = (detector.coordinate[0:3, 1, 0] - detector.coordinate[0:3, 0, 0]) / detector.resolution[0]
        yUnitVector = (detector.coordinate[0:3, 0, 1] - detector.coordinate[0:3, 0, 0]) / detector.resolution[1]

        backProjectedData = np.zeros((image.dimension[0] * image.dimension[1] * image.dimension[2]), dtype=np.float32)
        backProjectedData = np.ndarray.flatten(backProjectedData)
        normalVectorToDetectorPlane = np.array(normalVectorToDetectorPlane, dtype=np.float32)
        xUnitVector = np.array(xUnitVector, dtype=np.float32)
        yUnitVector = np.array(yUnitVector, dtype=np.float32)
        betaVector = np.array(betaVector, dtype=np.float32)

        # 分配 GPU 内存
        d_imageDimension = drv.mem_alloc(imageDimension.size * imageDimension.itemsize)
        d_imageResolution = drv.mem_alloc(imageResolution.size * imageResolution.itemsize)
        d_imageCoordinate = drv.mem_alloc(imageCoordinate.size * imageCoordinate.itemsize)
        d_sourceCoordinate = drv.mem_alloc(sourceCoordinate.size * sourceCoordinate.itemsize)
        d_detectorDimension = drv.mem_alloc(detectorDimension.size * detectorDimension.itemsize)
        d_detectorCoordinate = drv.mem_alloc(detectorCoordinate.size * detectorCoordinate.itemsize)
        d_detectorResolution = drv.mem_alloc(detectorResolution.size * detectorResolution.itemsize)
        d_normalVectorToDetectorPlane = drv.mem_alloc(normalVectorToDetectorPlane.size * normalVectorToDetectorPlane.itemsize)
        d_xUnitVector = drv.mem_alloc(xUnitVector.size * xUnitVector.itemsize)
        d_yUnitVector = drv.mem_alloc(yUnitVector.size * yUnitVector.itemsize)
        d_betaVector = drv.mem_alloc(betaVector.size * betaVector.itemsize)
        d_projectionView = drv.mem_alloc(projectionView1D.size * projectionView1D.itemsize)
        d_backProjectedData = drv.mem_alloc(backProjectedData.size * backProjectedData.itemsize)


        # 传递数据到 GPU
        drv.memcpy_htod(d_imageDimension, imageDimension)
        drv.memcpy_htod(d_imageResolution, imageResolution)
        drv.memcpy_htod(d_imageCoordinate, imageCoordinate)
        drv.memcpy_htod(d_sourceCoordinate, sourceCoordinate)
        drv.memcpy_htod(d_detectorDimension, detectorDimension)
        drv.memcpy_htod(d_detectorCoordinate, detectorCoordinate)
        drv.memcpy_htod(d_detectorResolution, detectorResolution)
        drv.memcpy_htod(d_normalVectorToDetectorPlane, normalVectorToDetectorPlane)
        drv.memcpy_htod(d_xUnitVector, xUnitVector)
        drv.memcpy_htod(d_yUnitVector, yUnitVector)
        drv.memcpy_htod(d_betaVector, betaVector)
        drv.memcpy_htod(d_projectionView, projectionView1D)
        

        # 设置 CUDA 网格和块的大小
        block_size = (8, 8, 8)
        grid_size = (int(np.ceil(image.dimension[0] / block_size[0])),
                    int(np.ceil(image.dimension[1] / block_size[1])),
                    int(np.ceil(image.dimension[2] / block_size[2])))
        tStartcudafun = time.time()
        # 调用 CUDA 核函数
        drv.Context.synchronize()
        backproject_func(d_imageDimension, d_imageResolution, d_imageCoordinate,
                        d_sourceCoordinate, np.float32(sourceSAD),
                        d_detectorDimension, d_detectorCoordinate, d_detectorResolution,
                        d_normalVectorToDetectorPlane, d_xUnitVector, d_yUnitVector,
                        d_betaVector, np.float32(dbeta),
                        d_projectionView, d_backProjectedData,
                        block=block_size, grid=grid_size)
        tEndcudafun = time.time()
        # 将结果从 GPU 复制回 CPU
        drv.Context.synchronize()
        drv.memcpy_dtoh(backProjectedData, d_backProjectedData)
        # 释放 GPU 内存
        d_imageDimension.free()
        d_imageResolution.free()
        d_imageCoordinate.free()
        d_sourceCoordinate.free()
        d_detectorDimension.free()
        d_detectorCoordinate.free()
        d_detectorResolution.free()
        d_normalVectorToDetectorPlane.free()
        d_xUnitVector.free()
        d_yUnitVector.free()
        d_betaVector.free()
        d_projectionView.free()
        d_backProjectedData.free()
        context.pop()
        tEndcuda = time.time()
        print('fbp cuda time is'+str(tEndcuda-tStartcuda)+' sec.')
        #print('cudafun time is'+str(tEndcudafun-tStartcudafun)+' sec.')
        # 将结果重塑为正确的形状
        backProjectedData = np.array(backProjectedData[:]).reshape((image.dimension[0], image.dimension[1], image.dimension[2]))

    
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


    return backProjectedData


def GetResult(result):
    global results
    results.append(result)


def BackProjection(projectionData, image, source, detector):

    reconImageData = np.zeros([image.dimension[0], image.dimension[1], image.dimension[2]])

    if Config.evenlyDistributedView:
        nView = Config.nView
        viewSet = np.arange(nView) * Config.deltaLamda 
    else:
        nView = len(Config.viewSet)
        viewSet = Config.viewSet

    if nView != np.shape(projectionData)[2]:
        sys.exit("nView in Config.py different from projection data")

    if Config.multiprocessing:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        for iView in range(nView):
            pool.apply_async(BackProjectView, args=(iView, viewSet, projectionData[:, :, iView], image, source, detector, 0), callback=GetResult)
        pool.close()
        pool.join()

        for iView in range(nView):
            reconImageData = reconImageData + np.array(results[iView][1])
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()
        source_trajectory = []
        detector_center_traj = []
        for iView in range(nView):
            result = BackProjectView(detector_center_traj,source_trajectory,fig,ax,iView, viewSet, projectionData[:, :, iView], image, source, detector, 0)
            print(np.max(result))
            print(np.max(projectionData[:, :, iView]))
            reconImageData += result
        plt.ioff()
        plt.show(block=True)

    return reconImageData
