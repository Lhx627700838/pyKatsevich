"""
forward projection from image
"""
import numpy as np
#from numba import cuda, float32, int32
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import Configuration.Config as Config
import multiprocessing
import ctypes
import sys
import math
import time




def SortFirst(val):
    return val[0]


def CalcRayDistance(startCoordinate, endCoordinate):
    return np.sqrt(np.sum(np.square(endCoordinate - startCoordinate)))


def RayTracing(imageCoordinate, imageResolution, sourceCoordinate, detectorCoordinate):

    if Config.rayTracingMethod == 'distance':
        imageWeight = RayTracingDistanceBased(imageCoordinate, imageResolution, sourceCoordinate, detectorCoordinate)
    elif Config.rayTracingMethod == 'sampling':
        imageWeight = RayTracingSamplingBased(imageCoordinate, imageResolution, sourceCoordinate, detectorCoordinate)

    return imageWeight


def RayTracingDistanceBased(imageCoordinate, imageResolution, sourceCoordinate, detectorCoordinate):
    [imageDimension[0], imageDimension[1], imageDimension[2]] = np.shape(imageCoordinate)[1:4]

    #imageCoordinate = np.round(imageCoordinate, 8)
    #sourceCoordinate = np.round(sourceCoordinate, 8)
    #detectorCoordinate = np.round(detectorCoordinate, 8)

    # imageWeight stores the length of the ray traversing through each voxel
    imageWeight = np.zeros([imageDimension[0], imageDimension[1], imageDimension[2]])

    xPlanes = np.concatenate((imageCoordinate[0, :, 0, 0] - imageResolution[0] / 2, [imageCoordinate[0, -1, 0, 0] + imageResolution[0] / 2]))
    yPlanes = np.concatenate((imageCoordinate[1, 0, :, 0] - imageResolution[1] / 2, [imageCoordinate[1, 0, -1, 0] + imageResolution[1] / 2]))
    zPlanes = np.concatenate((imageCoordinate[2, 0, 0, :] - imageResolution[2] / 2, [imageCoordinate[2, 0, 0, -1] + imageResolution[2] / 2]))

    # find intercept of source-detector line with x-, y- and z- planes

    if (detectorCoordinate[0] != sourceCoordinate[0]):
        gradVectorX = (xPlanes - sourceCoordinate[0]) / (detectorCoordinate[0] - sourceCoordinate[0])
        yCoordVectorX = sourceCoordinate[1] + gradVectorX * (detectorCoordinate[1] - sourceCoordinate[1])
        zCoordVectorX = sourceCoordinate[2] + gradVectorX * (detectorCoordinate[2] - sourceCoordinate[2])
        coordPlaneX = np.array([xPlanes, yCoordVectorX, zCoordVectorX])
    else:
        coordPlaneX = [[],[],[]]

    if (detectorCoordinate[1] != sourceCoordinate[1]):
        gradVectorY = (yPlanes - sourceCoordinate[1]) / (detectorCoordinate[1] - sourceCoordinate[1])
        xCoordVectorY = sourceCoordinate[0] + gradVectorY * (detectorCoordinate[0] - sourceCoordinate[0])
        zCoordVectorY = sourceCoordinate[2] + gradVectorY * (detectorCoordinate[2] - sourceCoordinate[2])
        coordPlaneY = np.array([xCoordVectorY, yPlanes, zCoordVectorY])
    else:
        coordPlaneY = [[],[],[]]

    if (detectorCoordinate[2] != sourceCoordinate[2]):
        gradVectorZ = (zPlanes - sourceCoordinate[2]) / (detectorCoordinate[2] - sourceCoordinate[2])
        xCoordVectorZ = sourceCoordinate[0] + gradVectorZ * (detectorCoordinate[0] - sourceCoordinate[0])
        yCoordVectorZ = sourceCoordinate[1] + gradVectorZ * (detectorCoordinate[1] - sourceCoordinate[1])
        coordPlaneZ = np.array([xCoordVectorZ, yCoordVectorZ, zPlanes])
    else:
        coordPlaneZ = [[],[],[]]

    # All unique intercept points, sorted by x-coordinate, y-coordinate or z-coordinate

    coordAllPlanes = np.swapaxes(np.round(np.concatenate((coordPlaneX, coordPlaneY, coordPlaneZ), axis=1), 8), 0, 1)
    coordAllPlanes = np.array(list(set(map(tuple, coordAllPlanes))))

    if (detectorCoordinate[0] != sourceCoordinate[0]):
        coordAllPlanes = np.array(sorted(coordAllPlanes, key=lambda l: l[0]))
    else:
        coordAllPlanes = np.array(sorted(coordAllPlanes, key=lambda l: l[1]))

    nIntercept = np.shape(coordAllPlanes)[0]

    for iIntercept in range(nIntercept-1):

        startCoordinate = coordAllPlanes[iIntercept, :]
        endCoordinate = coordAllPlanes[iIntercept+1, :]
        middleCoordinate = (startCoordinate + endCoordinate) / 2

        xCoordVoxel = np.round((middleCoordinate[0] - imageCoordinate[0, 0, 0, 0]) / imageResolution[0])
        yCoordVoxel = np.round((middleCoordinate[1] - imageCoordinate[1, 0, 0, 0]) / imageResolution[1])
        zCoordVoxel = np.round((middleCoordinate[2] - imageCoordinate[2, 0, 0, 0]) / imageResolution[2])

        if (xCoordVoxel >= 0) and (xCoordVoxel < imageDimension[0]) and (yCoordVoxel >= 0) and (yCoordVoxel < imageDimension[1]) and (zCoordVoxel >= 0) and (zCoordVoxel < imageDimension[2]):
            rayLength = CalcRayDistance(startCoordinate, endCoordinate)
            imageWeight[int(xCoordVoxel), int(yCoordVoxel), int(zCoordVoxel)] = rayLength

    return imageWeight


def RayTracingSamplingBased(imageCoordinate, imageResolution, sourceCoordinate, detectorCoordinate):

    [imageDimension[0], imageDimension[1], imageDimension[2]] = np.shape(imageCoordinate)[1:4]

    imageCoordinate = np.round(imageCoordinate, 8)
    sourceCoordinate = np.round(sourceCoordinate, 8)
    detectorCoordinate = np.round(detectorCoordinate, 8)

    # imageWeight stores the length of the ray traversing through each voxel
    imageWeight = np.zeros([imageDimension[0], imageDimension[1], imageDimension[2]])

    sourceDetectorDistance = CalcRayDistance(sourceCoordinate, detectorCoordinate)

    # compute the first and last plane of intersection between image volume and source-detector line for line sampling
    if abs(sourceCoordinate[0] - detectorCoordinate[0]) < abs(sourceCoordinate[1] - detectorCoordinate[1]):
        yBoundaryPlanes = [imageCoordinate[1, 0, 0, 0], imageCoordinate[1, 0, -1, 0]]
        distanceBound = (yBoundaryPlanes - sourceCoordinate[1]) / (detectorCoordinate[1] - sourceCoordinate[1]) * sourceDetectorDistance
    else:
        xBoundaryPlanes = [imageCoordinate[0, 0, 0, 0], imageCoordinate[0, -1, 0, 0]]
        distanceBound = (xBoundaryPlanes - sourceCoordinate[0]) / (detectorCoordinate[0] - sourceCoordinate[0]) * sourceDetectorDistance

    sampleDistanceStart = max(min(distanceBound) - np.sqrt(2) * max(imageResolution), 0)
    sampleDistanceEnd = min(max(distanceBound) + np.sqrt(2) * max(imageResolution), sourceDetectorDistance)

    sourceImageDistance = sampleDistanceStart
    while sourceImageDistance < sampleDistanceEnd:

        distanceRatio = sourceImageDistance / sourceDetectorDistance
        samplePointCoordinate = sourceCoordinate + distanceRatio * (detectorCoordinate - sourceCoordinate)

        xCoordVoxel = np.round((samplePointCoordinate[0] - imageCoordinate[0, 0, 0, 0]) / imageResolution[0])
        yCoordVoxel = np.round((samplePointCoordinate[1] - imageCoordinate[1, 0, 0, 0]) / imageResolution[1])
        zCoordVoxel = np.round((samplePointCoordinate[2] - imageCoordinate[2, 0, 0, 0]) / imageResolution[2])

        if (xCoordVoxel >= 0) and (xCoordVoxel < imageDimension[0]) and (yCoordVoxel >= 0) and (yCoordVoxel < imageDimension[1]) and (zCoordVoxel >= 0) and (zCoordVoxel < imageDimension[2]):
            imageWeight[int(xCoordVoxel), int(yCoordVoxel), int(zCoordVoxel)] = imageWeight[int(xCoordVoxel), int(yCoordVoxel), int(zCoordVoxel)] + 1

        sourceImageDistance = sourceImageDistance + Config.rayTracingSampleInterval

    return imageWeight

#向前投影传递图像、光源、探测器位置
def ForwardProject(image, source, detector, cuda_id):


    # C projector runs on Linux platform
    if Config.projectorPlatform in ['c', 'python']:
        forwardProjectHandle = ctypes.CDLL('./Projection/ForwardProject.so')
        forwardProjectHandle.ForwardProjectFunc.argtypes = (ctypes.POINTER(ctypes.c_int),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_int),
                                                            ctypes.POINTER(ctypes.c_float),
                                                            ctypes.POINTER(ctypes.c_float))

        #forwardProjectHandle.ForwardProjectFunc.restypes = ctypes.c_int

        detectorCoordinate = detector.coordinate
        detectorCoordinate = np.ndarray.flatten(detectorCoordinate)

        imageCoordinate = image.voxelCoordinate
        imageCoordinate = np.ndarray.flatten(imageCoordinate)

        imageData = image.data
        imageData = np.ndarray.flatten(imageData)

        arrayTypeImageDimension = ctypes.c_int * 3
        arrayTypeImageResolution = ctypes.c_float * 3
        arrayTypeImageCoordinate = ctypes.c_float * len(imageCoordinate)
        arrayTypeImageData = ctypes.c_float * len(imageData)
        arrayTypeSourceCoordinate = ctypes.c_float * 3
        arrayTypeDetectorDimension = ctypes.c_int * len(detector.dimension)
        arrayTypeDetectorCoordinate = ctypes.c_float * len(detectorCoordinate)

        forwardProjectedDataInit = list(np.zeros(detector.dimension[0] * detector.dimension[1]).astype(float))  # Create List with underlying memory
        forwardProjectedDataPointer = (ctypes.c_float * len(forwardProjectedDataInit))(*forwardProjectedDataInit)  # Create ctypes pointer to underlying memory
        #print(detectorCoordinate)
        forwardProjectHandle.ForwardProjectFunc(arrayTypeImageDimension(*image.dimension),
                                                arrayTypeImageResolution(*image.resolution),
                                                arrayTypeImageCoordinate(*imageCoordinate),
                                                arrayTypeImageData(*imageData),
                                                arrayTypeSourceCoordinate(*source.coordinate),
                                                arrayTypeDetectorDimension(*detector.dimension),
                                                arrayTypeDetectorCoordinate(*detectorCoordinate),
                                                forwardProjectedDataPointer)

        forwardProjectedData = np.array(forwardProjectedDataPointer[:]).reshape((detector.dimension[0], detector.dimension[1]))
        #print(forwardProjectedData[64, 64])

    elif Config.projectorPlatform in ['cuda']:
        tStartcuda = time.time()
        try:
            #print("CUDA initialized.")
            device = drv.Device(cuda_id)  # Define device variable
            context = device.make_context()
            #print(f"Forwardp Running on device: {cuda_id}, {device.name()}")
        except Exception as e:
            print(f"Error initializing CUDA: {e}")

        # Using PyCUDA for CUDA calls
        cuda_code = """
            extern __shared__ float sharedPlanes[]; // Dynamic shared memory allocation
            __device__ __constant__ float xPlanes[1024];
            __device__ __constant__ float yPlanes[1024];
            __device__ __constant__ float zPlanes[1024];
            
            __device__ void swap(float *xp, float *yp) {
                float temp = *xp;
                *xp = *yp;
                *yp = temp;
            }

            __device__ void selectionSort(float arr1[], float arr2[], float arr3[], int n) {
                int i, j, minIndex;
                for (i = 0; i < n - 1; i++) {
                    minIndex = i;
                    for (j = i + 1; j < n; j++) {
                        if (arr1[j] < arr1[minIndex]) {
                            minIndex = j;
                        }
                    }
                    swap(&arr1[minIndex], &arr1[i]);
                    swap(&arr2[minIndex], &arr2[i]);
                    swap(&arr3[minIndex], &arr3[i]);
                }
            }

            __device__ void removeDuplicate(float allInterceptX[], float allInterceptY[], float allInterceptZ[], int &nAllIntercept) {
                int i, j, k;
                for (i = 0; i < nAllIntercept; i++) {
                    for (j = i + 1; j < nAllIntercept; j++) {
                        if ((allInterceptX[i] == allInterceptX[j]) && (allInterceptY[i] == allInterceptY[j]) && (allInterceptZ[i] == allInterceptZ[j])) {
                            for (k = j; k < nAllIntercept - 1; k++) {
                                allInterceptX[k] = allInterceptX[k + 1];
                                allInterceptY[k] = allInterceptY[k + 1];
                                allInterceptZ[k] = allInterceptZ[k + 1];
                            }
                            nAllIntercept--;
                            j--;
                        }
                    }
                }
            }

            __device__ float CalcRayDistance(const float startCoordinate[], const float endCoordinate[]) {
                float sumSquared = 0;
                for (int i = 0; i < 3; i++) {
                    sumSquared += powf((endCoordinate[i] - startCoordinate[i]), 2);
                }
                return sqrtf(sumSquared);
            }

            __device__ float RayTracingDistanceBased(const int imageDimension[], const float imageCoordinate[], const float imageResolution[], const float imageData[], const float sourceCoordinate[], const float detectorCoordinateThis[], 
                                                float *sharedInterceptX, float *sharedInterceptY, float *sharedInterceptZ, int threadIdxFlat) {

                int nAllIntercept = 0;
                if (detectorCoordinateThis[0] != sourceCoordinate[0]) nAllIntercept += imageDimension[0] + 1;
                if (detectorCoordinateThis[1] != sourceCoordinate[1]) nAllIntercept += imageDimension[1] + 1;
                if (detectorCoordinateThis[2] != sourceCoordinate[2]) nAllIntercept += imageDimension[2] + 1;

                int interceptCounter = 0;
                int localIndex = blockDim.x * threadIdx.x + threadIdx.y;
                int flag = 0;
                int flagz = 0;
                if (fabs(detectorCoordinateThis[0] - sourceCoordinate[0]) > 100) {
                    for (int iX = 0; iX <= imageDimension[0]; iX++) {
                        float gradVectorX = (xPlanes[iX] - sourceCoordinate[0]) / (detectorCoordinateThis[0] - sourceCoordinate[0]);
                        if (interceptCounter < blockDim.x * blockDim.y * 1024) {
                            sharedInterceptX[interceptCounter + localIndex * 1024] = xPlanes[iX];
                            sharedInterceptY[interceptCounter + localIndex * 1024] = sourceCoordinate[1] + gradVectorX * (detectorCoordinateThis[1] - sourceCoordinate[1]);
                            sharedInterceptZ[interceptCounter + localIndex * 1024] = sourceCoordinate[2] + gradVectorX * (detectorCoordinateThis[2] - sourceCoordinate[2]);
                            //printf("x point :%f, y point: %f, z point:%f  ", *(sharedInterceptX+interceptCounter + localIndex * 1024),*(sharedInterceptY+interceptCounter + localIndex * 1024),*(sharedInterceptZ+interceptCounter + localIndex * 1024));
                            interceptCounter++;
                           
                        }
                    }
                    flag = 1;
                }

                __syncthreads();
                
                if (fabs(detectorCoordinateThis[1] - sourceCoordinate[1]) > 100 && flag ==0) {
                    for (int iY = 0; iY <= imageDimension[1]; iY++) {
                        float gradVectorY = (yPlanes[iY] - sourceCoordinate[1]) / (detectorCoordinateThis[1] - sourceCoordinate[1]);
                        if (interceptCounter < blockDim.x * blockDim.y * 1024) {
                            sharedInterceptX[interceptCounter + localIndex * 1024] = sourceCoordinate[0] + gradVectorY * (detectorCoordinateThis[0] - sourceCoordinate[0]);
                            sharedInterceptY[interceptCounter + localIndex * 1024] = yPlanes[iY];
                            sharedInterceptZ[interceptCounter + localIndex * 1024] = sourceCoordinate[2] + gradVectorY * (detectorCoordinateThis[2] - sourceCoordinate[2]);
                            interceptCounter++;
                        }
                    }
                    flag = 1;
                }

                __syncthreads();

                if (detectorCoordinateThis[2] != sourceCoordinate[2] && flag ==0) {
                    for (int iZ = 0; iZ <= imageDimension[2]; iZ++) {
                        float gradVectorZ = (zPlanes[iZ] - sourceCoordinate[2]) / (detectorCoordinateThis[2] - sourceCoordinate[2]);
                        if (interceptCounter < blockDim.x * blockDim.y * 1024) {
                            sharedInterceptX[interceptCounter + localIndex * 1024] = sourceCoordinate[0] + gradVectorZ * (detectorCoordinateThis[0] - sourceCoordinate[0]);
                            sharedInterceptY[interceptCounter + localIndex * 1024] = sourceCoordinate[1] + gradVectorZ * (detectorCoordinateThis[1] - sourceCoordinate[1]);
                            sharedInterceptZ[interceptCounter + localIndex * 1024] = zPlanes[iZ];
                            interceptCounter++;
                        }
                    }
                    flagz = 1;
                }

                __syncthreads();
                //removeDuplicate(sharedInterceptX+ localIndex * 1024, sharedInterceptY+ localIndex * 1024, sharedInterceptZ+ localIndex * 1024, nAllIntercept);
                //if (detectorCoordinateThis[0] != sourceCoordinate[0]) {
                //    selectionSort(sharedInterceptX+ localIndex * 2048, sharedInterceptY+ localIndex * 2048, sharedInterceptZ+ localIndex * 2048, nAllIntercept);
                    //printf("XXXXXXXXXXXXXXXXXXX");
                //} else {
                //    selectionSort(sharedInterceptY+ localIndex * 2048, sharedInterceptX+ localIndex * 2048, sharedInterceptZ+ localIndex * 2048, nAllIntercept);
                    //printf("YYYYYYYYYYYYYYYYYYYYYYYYYYYY");
                //}

                float projSum = 0.0;
                int limit = imageDimension[0];
                if (flagz == 1) limit = imageDimension[2];
                for (int iIntercept = 0; iIntercept < (limit-2); iIntercept++) {
                    float startCoordinate[3], endCoordinate[3], middleCoordinate[3];
                    int xCoordVoxel, yCoordVoxel, zCoordVoxel;
                    float rayLength;

                    startCoordinate[0] = sharedInterceptX[iIntercept + localIndex * 1024];
                    startCoordinate[1] = sharedInterceptY[iIntercept + localIndex * 1024];
                    startCoordinate[2] = sharedInterceptZ[iIntercept + localIndex * 1024];

                    endCoordinate[0] = sharedInterceptX[iIntercept + 1 + localIndex * 1024];
                    endCoordinate[1] = sharedInterceptY[iIntercept + 1 + localIndex * 1024];
                    endCoordinate[2] = sharedInterceptZ[iIntercept + 1 + localIndex * 1024];
                    //printf("start x:%f,y:%f,z:%f, end x:%f,y:%f,z:%f   ",startCoordinate[0],startCoordinate[1],startCoordinate[2],endCoordinate[0],endCoordinate[1],endCoordinate[2]);
                    middleCoordinate[0] = (startCoordinate[0] + endCoordinate[0]) / 2;
                    middleCoordinate[1] = (startCoordinate[1] + endCoordinate[1]) / 2;
                    middleCoordinate[2] = (startCoordinate[2] + endCoordinate[2]) / 2;

                    xCoordVoxel = round((middleCoordinate[0] - imageCoordinate[0]) / imageResolution[0]);
                    yCoordVoxel = round((middleCoordinate[1] - imageCoordinate[1]) / imageResolution[1]);
                    zCoordVoxel = round((middleCoordinate[2] - imageCoordinate[2]) / imageResolution[2]);
                    

                    if (xCoordVoxel >= 0 && xCoordVoxel < imageDimension[0] && yCoordVoxel >= 0 && yCoordVoxel < imageDimension[1] && zCoordVoxel >= 0 && zCoordVoxel < imageDimension[2]) {
                        rayLength = CalcRayDistance(startCoordinate, endCoordinate);
                        if (rayLength > 80) printf("start x:%f,y:%f,z:%f, end x:%f,y:%f,z:%f and distance:%f   %d    ",startCoordinate[0],startCoordinate[1],startCoordinate[2],endCoordinate[0],endCoordinate[1],endCoordinate[2],rayLength,flagz);
                        
                        projSum += rayLength * imageData[zCoordVoxel + imageDimension[2] * (yCoordVoxel + imageDimension[1] * xCoordVoxel)];
                        //printf("voxel x:%d,y:%d,z:%d and data:%f    ",xCoordVoxel,yCoordVoxel,zCoordVoxel,projSum);
                        
                        
                    }
                }
                //printf("xID:%d, yID: %d ,projSum: %f ", blockIdx.x * blockDim.x + threadIdx.x,blockIdx.y * blockDim.y + threadIdx.y,projSum);
                return projSum;
            }

            __global__ void ForwardProjectFunc(const int *imageDimension, const float *imageResolution, const float *imageCoordinate, const float *imageData,
                                            const float *sourceCoordinate, const int *detectorDimension, const float *detectorCoordinate, 
                                            float *forwardProjectedData) {
                extern __shared__ float sharedData[];
                float *sharedInterceptX = sharedData;
                float *sharedInterceptY = &sharedInterceptX[blockDim.x * blockDim.y * 1024];
                float *sharedInterceptZ = &sharedInterceptY[blockDim.x * blockDim.y * 1024];
                
                float detectorCoordinateThis[3];
                int iX = blockIdx.x * blockDim.x + threadIdx.x;  // Global X index
                int iY = blockIdx.y * blockDim.y + threadIdx.y;  // Global Y index
                
                if (iX >= detectorDimension[0] || iY >= detectorDimension[1]) {
                    return;
                }
                int threadIdxFlat = iX * detectorDimension[1] + iY;
                //printf("imageDimension: %d, imageResolution: %f, imageCoordinate: %f, imageData: %f, sourceCoordinate: %f, detectorDimension: %d, detectorCoordinate: %f",*imageDimension, *imageResolution, *imageCoordinate, *imageData, *sourceCoordinate, *detectorDimension, *detectorCoordinate);
                for (int iCoord = 0; iCoord < 3; iCoord++) {   
                    detectorCoordinateThis[iCoord] = detectorCoordinate[iY + detectorDimension[1] * (iX + detectorDimension[0] * iCoord)];
                }
                forwardProjectedData[threadIdxFlat] = RayTracingDistanceBased(imageDimension, imageCoordinate, imageResolution, imageData, sourceCoordinate, detectorCoordinateThis, 
                                                                            sharedInterceptX, sharedInterceptY, sharedInterceptZ, threadIdxFlat);
                //printf("IDflat:%d,data:%f   ",threadIdxFlat,forwardProjectedData[threadIdxFlat]);
                __syncthreads();
            }
        """
        tStartcudafun = time.time()
        mod = SourceModule(cuda_code)
        
        ray_tracing_func = mod.get_function("ForwardProjectFunc")

        # Memory Allocation and Transfer to GPU (Optimized)
        imageDimension = np.array(image.dimension, dtype=np.int32)
        imageResolution = np.array(image.resolution, dtype=np.float32)
        imageCoordinate = np.ndarray.flatten(image.voxelCoordinate).astype(np.float32)
        imageData = np.ndarray.flatten(image.data).astype(np.float32)
        sourceCoordinate = np.array(source.coordinate, dtype=np.float32)
        detectorDimension = np.array(detector.dimension, dtype=np.int32)
        forwardProjectedData = np.zeros(detector.dimension[0] * detector.dimension[1], dtype=np.float32)
        detectorCoordinate = np.ndarray.flatten(detector.coordinate).astype(np.float32)
        xPlanes = np.array([(i - 0.5 * imageDimension[0]) for i in range(imageDimension[0])], dtype=np.float32)
        yPlanes = np.array([(i - 0.5 * imageDimension[1]) for i in range(imageDimension[1])], dtype=np.float32)
        zPlanes = np.array([(i - 0.5 * imageDimension[2]) for i in range(imageDimension[2])], dtype=np.float32)
        xPlanes_gpu, _ = mod.get_global('xPlanes')
        yPlanes_gpu, _ = mod.get_global('yPlanes')
        zPlanes_gpu, _ = mod.get_global('zPlanes')  


        d_imageDimension = drv.mem_alloc(imageDimension.size * imageDimension.itemsize)
        d_imageResolution = drv.mem_alloc(imageResolution.size * imageResolution.itemsize)
        d_imageCoordinate = drv.mem_alloc(imageCoordinate.size * imageCoordinate.itemsize)
        d_imageData = drv.mem_alloc(imageData.size * imageData.itemsize)
        d_sourceCoordinate = drv.mem_alloc(sourceCoordinate.size * sourceCoordinate.itemsize)
        d_detectorDimension = drv.mem_alloc(detectorDimension.size * detectorDimension.itemsize)
        d_detectorCoordinate = drv.mem_alloc(detectorCoordinate.size * detectorCoordinate.itemsize)
        d_forwardProjectedData = drv.mem_alloc(forwardProjectedData.size * forwardProjectedData.itemsize)
        drv.memcpy_htod(xPlanes_gpu, xPlanes)
        drv.memcpy_htod(yPlanes_gpu, yPlanes)
        drv.memcpy_htod(zPlanes_gpu, zPlanes)

        drv.memcpy_htod(d_detectorCoordinate, detectorCoordinate)
        drv.memcpy_htod(d_imageDimension, imageDimension)
        drv.memcpy_htod(d_imageResolution, imageResolution)
        drv.memcpy_htod(d_imageCoordinate, imageCoordinate)
        drv.memcpy_htod(d_imageData, imageData)
        drv.memcpy_htod(d_sourceCoordinate, sourceCoordinate)
        drv.memcpy_htod(d_detectorDimension, detectorDimension)

        block_size = (2, 2, 1)
        grid_size = (int(np.ceil(detectorDimension[0] / block_size[0])), int(np.ceil(detectorDimension[1] / block_size[1])), 1)
        shared_mem_size = 3 * 2 * 2 * 4 * 1024  # 3 arrays, each with blockDim.x * blockDim.y elements of float (4 bytes)
        #print(shared_mem_size) 
        if shared_mem_size > 48 * 1024: #最大共享内存为每个block 48KB
            raise ValueError("Shared memory size exceeds the device limit of 48 KB.")

        drv.Context.synchronize()
        ray_tracing_func(
            d_imageDimension, d_imageResolution, d_imageCoordinate, d_imageData,
            d_sourceCoordinate, d_detectorDimension, d_detectorCoordinate, 
            d_forwardProjectedData,
            block=block_size, grid=grid_size, shared=shared_mem_size)



        drv.Context.synchronize()
        tEndcudafun = time.time()
        drv.memcpy_dtoh(forwardProjectedData, d_forwardProjectedData)
        #print(forwardProjectedData)
        forwardProjectedData = np.array(forwardProjectedData[:]).reshape((detector.dimension[0], detector.dimension[1]))
        d_imageDimension.free()
        d_imageResolution.free()
        d_imageCoordinate.free()
        d_imageData.free()
        d_sourceCoordinate.free()
        d_detectorDimension.free()
        d_detectorCoordinate.free()
        d_forwardProjectedData.free()

        tEndcuda = time.time()
        print('cuda time is ' + str(tEndcuda - tStartcuda) + ' sec.')
        print('cudafun time is ' + str(tEndcudafun - tStartcudafun) + ' sec.')
        context.pop()
        #print(forwardProjectedData[64, 64])



    if Config.projectorPlatform in ['NULL']:
        forwardProjectedData = np.zeros([detector.dimension[0], detector.dimension[1]])

        sourceCoordinate = source.coordinate
        for iX in range(detector.dimension[0]):
            for iY in range(detector.dimension[1]):

                detectorCoordinate = detector.coordinate[0:3, iX, iY]

                #rayTracing through image
                imageWeight = RayTracing(image.voxelCoordinate, image.resolution, sourceCoordinate, detectorCoordinate)
                forwardProjectedData[iX, iY] = np.sum(imageWeight * image.data)

    return forwardProjectedData


def ForwardProjectView(iView, viewSet, image, source, detector, cuda_id):
    
    #调用 source 对象的 ComputeSourceCoordinate 方法来计算光源的坐标。传递给该方法的参数包括源到探测器的距离 SAD、源的轴角 sourceAxisAngle 以及当前视图的角度 viewSet[iView]
    source.ComputeSourceCoordinate(source.SAD, source.sourceAxisAngle, viewSet[iView])
    print(f"Starting ForwardProjectView for view {iView} on CUDA device {cuda_id}")
    
    #调用 detector 对象的 ComputeDetectCoordinate 方法来计算探测器的坐标。传递给该方法的参数包括探测器到源的距离 SAD、源到探测器的距离 SDD、探测器的尺寸 dimension、探测器的分辨率 resolution、探测器的偏移量 offset、探测器的轴角 detectorAxisAngle、探测器的倾斜角 detectorTiltAngle 以及当前视图的角度 viewSet[iView]
    detector.ComputeDetectCoordinate(detector.SAD, detector.SDD, detector.dimension, detector.resolution,
                                     detector.offset, detector.detectorAxisAngle, detector.detectorTiltAngle, viewSet[iView], cuda_id)

    #调用 ForwardProject 函数，传入 image、source 和 detector 作为参数，计算当前视图的前向投影结果，并将结果存储在 result 变量中
    result = ForwardProject(image, source, detector, cuda_id)
    #print(iView)
    return (iView, result)



def GetResult(result):
    global results
    results.append(result)


def ForwardProjection(image, source, detector):

    if Config.evenlyDistributedView:
        nView = Config.nView
        viewSet = np.arange(nView) * (360 / nView)
    else:
        nView = len(Config.viewSet)
        viewSet = Config.viewSet

    forwardProjectedData = np.zeros([Config.detectorDimension[0], Config.detectorDimension[1], nView])

    global results
    results = []

    if Config.multiprocessing:
        if Config.projectorPlatform == 'cuda':
            device_count = drv.Device.count()  # 获取GPU数量
            pool = multiprocessing.Pool(processes=device_count)  # 创建与GPU数量相同的进程池
            #print(device_count)
            #print(pool)
            # 分配每个GPU任务
            for index in range(math.ceil(Config.nView/device_count)):
                device_id = 0
                for device_id in range(device_count):
                    print('id: '+str(device_id))
                    pool.apply_async(ForwardProjectView, args=(device_id+index*device_count, viewSet, image, source, detector, device_id), callback=GetResult)
                    print(device_id+index*device_count)

            pool.close()
            pool.join()

            for iView in range(nView):
                #从 results 中获取视图索引。
                viewIndex = np.array(results[iView][0])
                #将每个视图的投影数据填入 forwardProjectedData 数组中
                forwardProjectedData[:, :, viewIndex] = np.array(results[iView][1])
        else:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            #print(pool)
            #创建一个进程池，进程数为 CPU 核心数
            for iView in range(nView):
                #异步地调用 ForwardProjectView 函数，并将结果通过 GetResult 回调处理。
                pool.apply_async(ForwardProjectView, args=(iView, viewSet, image, source, detector, 0), callback=GetResult)
                #关闭进程池，防止新的任务提交,等待所有进程完成任务
            pool.close()
            pool.join()
            #再次遍历所有视图
            for iView in range(nView):
                #从 results 中获取视图索引。
                viewIndex = np.array(results[iView][0])
                #将每个视图的投影数据填入 forwardProjectedData 数组中
                forwardProjectedData[:, :, viewIndex] = np.array(results[iView][1])
    else:
        for iView in range(nView):
            #source.ComputeSourceCoordinate(source.SAD, source.sourceAxisAngle, viewSet[iView])
            #detector.ComputeDetectCoordinate(detector.SAD, detector.SDD, detector.dimension, detector.resolution,
            #                                 detector.offset, detector.detectorAxisAngle, detector.detectorTiltAngle, viewSet[iView])
            GetResult(ForwardProjectView(iView, viewSet, image, source, detector, 0))
            #print(iView)

        for iView in range(nView):
            viewIndex = np.array(results[iView][0])
            forwardProjectedData[:, :, viewIndex] = np.array(results[iView][1])

    return forwardProjectedData