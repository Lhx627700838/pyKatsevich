'''
“滤波后的投影数据 → 每个角度做 cone-beam 反投影 → 用 Katsevich 权重因子积分 → 得到最终重建图像”
'''
import numpy as np
def find_closest_lambda(x, y, z, R, P, lambdas):
    min_dist = float('inf')
    min_lambda = 0
    for lam in lambdas:
        x_s = R * np.cos(lam)
        y_s = R * np.sin(lam)
        z_s = P * lam / (2 * np.pi)
        dist = (x - x_s)**2 + (y - y_s)**2 + (z - z_s)**2
        if dist < min_dist:
            min_dist = dist
            min_lambda = lam
    return min_lambda

def find_pi_line_range(x, y, z, lambdas, R, P):
    lam_m = find_closest_lambda(x, y, z, R, P, lambdas)
    lam_i = lam_m - np.pi / 2
    lam_o = lam_m + np.pi / 2
    # Clip to available λ range
    lam_i = max(lambdas[0], lam_i)
    lam_o = min(lambdas[-1], lam_o)
    # 返回索引范围
    valid_indices = np.where((lambdas >= lam_i) & (lambdas <= lam_o))[0]
    return valid_indices
