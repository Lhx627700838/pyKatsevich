import numpy as np
from scipy.interpolate import interpn

# 构建原始半格采样坐标
lambda_half = np.arange(0.5, 4000-0.5, 1.0)  # 3999
alpha_half  = np.arange(0.5, 1376-0.5, 1.0)  # 1375
w_coords    = np.arange(0, 144, 1.0)        # 144

points = (lambda_half, w_coords, alpha_half)


# 定义目标整格坐标
lambda_full = np.arange(0, 4000, 1.0)
alpha_full  = np.arange(0, 1376, 1.0)
w_full      = np.arange(0, 144, 1.0)

xi = np.meshgrid(lambda_full, w_full, alpha_full, indexing='ij')

print(lambda_half)
print(lambda_full)

print(alpha_half)
print(alpha_full)

print(w_coords)
print(w_full)