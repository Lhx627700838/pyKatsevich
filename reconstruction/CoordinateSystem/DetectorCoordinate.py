"""
setup image coordinate system
"""
import numpy as np
from Configuration import Config
from TransformationMatrix import SetTransformationMatrix
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import time


class Detector:
    def __init__(self, SAD, SDD, dimension, resolution, offset, detectorTiltAngle, detectorAxisAngle):
        self.SAD = SAD
        self.SDD = SDD
        self.dimension = dimension
        self.resolution = resolution
        self.offset = offset
        self.detectorTiltAngle = detectorTiltAngle
        self.detectorAxisAngle = detectorAxisAngle
        self.coordinate = np.zeros([3, dimension[0], dimension[1]])


    def ComputeDetectCoordinate(self, SAD, SDD, dimension, resolution, offset, detectorAxisAngle, detectorTiltAngle, axialAngle, iVew, cuda_id):
        stride = iVew * Config.deltaZ
        if Config.projectorPlatform != 'cuda':
            tStart = time.time()
            detectorCoordinate = np.zeros([4, dimension[0], dimension[1]])
            for iX in range(dimension[0]):
                for iY in range(dimension[1]):
                    # define coordinates of the center of each detector pixel at origin
                    detectorCoordinate[0, iX, iY] = (iX - dimension[0] / 2 + 0.5) * resolution[0] + offset[0]
                    detectorCoordinate[1, iX, iY] = 0
                    detectorCoordinate[2, iX, iY] = (iY - dimension[1] / 2 + 0.5) * resolution[1] + offset[1]
                    detectorCoordinate[3, iX, iY] = 1

                    # detector tilt
                    detectorTiltTranformationMatrix = SetTransformationMatrix.DetectorTiltTransformation(detectorTiltAngle)
                    detectorCoordinate[:, iX, iY] = np.matmul(detectorTiltTranformationMatrix, detectorCoordinate[:, iX, iY])

                    # SDD
                    detectorCoordinate[1, iX, iY] = detectorCoordinate[1, iX, iY] + (SDD - SAD)

                    # source axis angle
                    sourceAxisTiltTransformationMatrix = SetTransformationMatrix.SourceAxisTiltTransformation(detectorAxisAngle)
                    detectorCoordinate[:, iX, iY] = np.matmul(sourceAxisTiltTransformationMatrix, detectorCoordinate[:, iX, iY])

                    # axial angle
                    axialAngleTransformationMatrix = SetTransformationMatrix.AxialAngleTransformation(axialAngle)
                    detectorCoordinate[:, iX, iY] = np.matmul(axialAngleTransformationMatrix, detectorCoordinate[:, iX, iY])

            self.coordinate = np.round(detectorCoordinate[0:3, :, :], 8)
            tEnd = time.time()
            print('coordinate time: ' + str(np.round(tEnd-tStart, 1)) + ' sec.')
        else:
            try:
                #print(f"Initializing CUDA device with ID: {cuda_id}")
                drv.init()
                device = drv.Device(cuda_id)  # 获取设备
                #print(f"Device {cuda_id} initialized: {device.name()}")
                context = device.make_context()  # 创建上下文
                #print(f"Context created for device {cuda_id}")
            except Exception as e:
                print(f"Error initializing CUDA: {e}")
            cuda_code = """
                __device__ void matmul(const float* matrix, float* coord) {
                    float result[4] = {0.0f};
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            result[i] += matrix[i * 4 + j] * coord[j];
                        }
                    }
                    for (int i = 0; i < 4; i++) {
                        coord[i] = result[i];
                    }
                }

                __global__ void compute_detector_coordinates(float *detectorCoordinate, int dimX, int dimY, float resolutionX, float resolutionY, float offsetX, float offsetY, 
                                                            float* detectorTiltMatrix, float SDD, float SAD, float* axisTiltMatrix, float* axialAngleMatrix) {
                    int iX = blockIdx.x * blockDim.x + threadIdx.x;
                    int iY = blockIdx.y * blockDim.y + threadIdx.y;

                    if (iX >= dimX || iY >= dimY)
                        return;

                    int index = iX * dimY + iY;

                    // Define coordinates of the center of each detector pixel at origin
                    float coord[4] = {
                        (iX - dimX / 2.0f + 0.5f) * resolutionX + offsetX,
                        0,
                        (iY - dimY / 2.0f + 0.5f) * resolutionY + offsetY,
                        1
                    };

                    // Apply detector tilt transformation
                    matmul(detectorTiltMatrix, coord);

                    // Adjust for SDD and SAD
                    coord[1] += (SDD - SAD);

                    // Apply source axis tilt transformation
                    matmul(axisTiltMatrix, coord);

                    // Apply axial angle transformation
                    matmul(axialAngleMatrix, coord);

                    // Write result to global memory
                    for (int i = 0; i < 4; i++) {
                        detectorCoordinate[i * dimX * dimY + index] = coord[i];
                    }
                }
                """

            mod = SourceModule(cuda_code, options=["-allow-unsupported-compiler"])
            compute_coordinates = mod.get_function("compute_detector_coordinates")

            detector_coordinate = np.zeros((4, dimension[0], dimension[1]), dtype=np.float32)
            

            # Prepare transformation matrices (you need to provide the matrices)
            detectorTiltMatrix = SetTransformationMatrix.DetectorTiltTransformation(detectorTiltAngle).flatten().astype(np.float32)
            axisTiltMatrix = SetTransformationMatrix.SourceAxisTiltTransformation(detectorAxisAngle).flatten().astype(np.float32)
            axialAngleMatrix = SetTransformationMatrix.AxialAngleTransformation(axialAngle).flatten().astype(np.float32)

            # Allocate memory on GPU
            detectorTiltMatrix_gpu = drv.mem_alloc(detectorTiltMatrix.nbytes)
            axisTiltMatrix_gpu = drv.mem_alloc(axisTiltMatrix.nbytes)
            axialAngleMatrix_gpu = drv.mem_alloc(axialAngleMatrix.nbytes)
            detector_coordinate_gpu = drv.mem_alloc(detector_coordinate.nbytes)

            # Copy matrices to GPU
            drv.memcpy_htod(detectorTiltMatrix_gpu, detectorTiltMatrix)
            drv.memcpy_htod(axisTiltMatrix_gpu, axisTiltMatrix)
            drv.memcpy_htod(axialAngleMatrix_gpu, axialAngleMatrix)

            block_size = (16, 16, 1)
            grid_size = (int(np.ceil(dimension[0] / block_size[0])), int(np.ceil(dimension[1] / block_size[1])), 1)

            tStart = time.time()

            compute_coordinates(
                detector_coordinate_gpu, np.int32(dimension[0]), np.int32(dimension[1]),
                np.float32(resolution[0]), np.float32(resolution[1]),
                np.float32(offset[0]), np.float32(offset[1]),
                detectorTiltMatrix_gpu, np.float32(SDD), np.float32(SAD),
                axisTiltMatrix_gpu, axialAngleMatrix_gpu,
                block=block_size, grid=grid_size
            )
            
            # Copy result back to CPU
            drv.memcpy_dtoh(detector_coordinate, detector_coordinate_gpu)
            context.pop()
            tEnd = time.time()
            #print('CUDA coordinate time:', round(tEnd - tStart, 8), 'sec.')
            detector_coordinate = np.array(detector_coordinate[:]).reshape((4, dimension[0], dimension[1]))
            self.coordinate = np.round(detector_coordinate[0:3, :, :], 8)
            self.coordinate[2,:,:] += stride
        

def SetupDetectorCoordinate():

    detector = Detector(Config.SAD, Config.SDD, Config.detectorDimension, Config.detectorResolution, Config.detectorOffset, Config.detectorTiltAngle, Config.detectorAxisAngle)

    return detector
