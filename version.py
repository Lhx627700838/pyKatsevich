import cupy
print(cupy.__version__)
print(cupy.cuda.runtime.getDeviceProperties(0)['name'])