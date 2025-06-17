import cupy as cp

def katsevich_backprojection_gpu(input_array_np, conf_dict, volume_shape, lambda_array_np):
    # 参数
    nx, ny, nz = volume_shape
    dx, dy, dz = conf_dict['delta_x'], conf_dict['delta_y'], conf_dict['delta_z']
    x_min, y_min, z_min = conf_dict['x_min'], conf_dict['y_min'], conf_dict['z_min']
    R = conf_dict['scan_radius']
    P = conf_dict['progress_per_turn']
    D = conf_dict['scan_diameter']
    pixel_spacing_u = conf_dict['detector_pixel_span_u']
    pixel_spacing_v = conf_dict['detector_pixel_span_v']
    num_angles = len(lambda_array_np)
    det_rows, det_cols = input_array_np.shape[1:]

    # 组织 conf 数组传给 GPU
    conf = cp.array([dx, dy, dz, x_min, y_min, z_min, R, P, D,
                     pixel_spacing_u, pixel_spacing_v, num_angles, det_rows, det_cols], dtype=cp.float32)

    # 转换为 CuPy 数据
    input_array_cp = cp.asarray(input_array_np, dtype=cp.float32)
    lambda_array_cp = cp.asarray(lambda_array_np, dtype=cp.float32)
    rec_volume_cp = cp.zeros((nx, ny, nz), dtype=cp.float32)

    # CUDA 网格配置
    threadsperblock = (4, 4, 4)
    blockspergrid_x = (nx + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (ny + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_z = (nz + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # 启动 GPU 核函数
    katsevich_backprojection_kernel[blockspergrid, threadsperblock](
        input_array_cp,
        rec_volume_cp,
        lambda_array_cp,
        conf,
        (nx, ny, nz)
    )

    # 拷贝回 CPU
    return cp.asnumpy(rec_volume_cp)

import numpy as np
from numba import cuda, float32
import math

@cuda.jit(device=True)
def find_closest_lambda_gpu(x, y, z, R, P, lambdas, num_lambdas):
    min_dist = 1e20
    min_lambda = 0.0
    for idx in range(num_lambdas):
        lam = lambdas[idx]
        x_s = R * math.cos(lam)
        y_s = R * math.sin(lam)
        z_s = P * lam / (2 * math.pi)
        dist = (x - x_s)**2 + (y - y_s)**2 + (z - z_s)**2
        if dist < min_dist:
            min_dist = dist
            min_lambda = lam
    return min_lambda

from numba import cuda, float32
import math

@cuda.jit(device=True)
def find_pi_line_lambda_gpu(x, y, z, R, P, lambdas, num_lambdas):
    """
    For a given point (x, y, z), find a lambda_i such that the pi-line connecting
    a(lambda_i) and a(lambda_i + pi) approximately passes through the point.

    Returns:
        lambda_i that best fits the point (or -1.0 if not found)
    """
    best_lambda = -1.0
    min_error = 1e10

    for idx in range(num_lambdas - 1):
        lam_i = lambdas[idx]
        lam_o = lam_i + math.pi

        if lam_o > lambdas[num_lambdas - 1]:
            continue

        # Source positions at λ_i and λ_o
        x_ai = R * math.cos(lam_i)
        y_ai = R * math.sin(lam_i)
        z_ai = P * lam_i / (2 * math.pi)

        x_ao = R * math.cos(lam_o)
        y_ao = R * math.sin(lam_o)
        z_ao = P * lam_o / (2 * math.pi)

        vx = x_ao - x_ai
        vy = y_ao - y_ai
        vz = z_ao - z_ai

        wx = x - x_ai
        wy = y - y_ai
        wz = z - z_ai

        v_dot_v = vx * vx + vy * vy + vz * vz
        v_dot_w = vx * wx + vy * wy + vz * wz

        t = v_dot_w / v_dot_v

        if t >= 0.0 and t <= 1.0:
            proj_x = x_ai + t * vx
            proj_y = y_ai + t * vy
            proj_z = z_ai + t * vz

            dx = proj_x - x
            dy = proj_y - y
            dz = proj_z - z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist < min_error:
                min_error = dist
                best_lambda = lam_i

    if min_error < 1.0:  # threshold in mm
        return best_lambda
    else:
        return -1.0  # no valid pi-line found


@cuda.jit
def katsevich_backprojection_kernel(
    input_array,  # (num_angles, det_rows, det_cols)
    rec_volume,
    lambda_array,
    conf,
    vol_shape
):
    nx, ny, nz = vol_shape
    dx, dy, dz = conf[0], conf[1], conf[2]
    x_min, y_min, z_min = conf[3], conf[4], conf[5]
    R = conf[6]
    P = conf[7]
    D = conf[8]
    pixel_spacing_u = conf[9]
    pixel_spacing_v = conf[10]
    num_angles = int(conf[11])
    det_rows = int(conf[12])
    det_cols = int(conf[13])

    i, j, k = cuda.grid(3)
    if i >= nx or j >= ny or k >= nz:
        return

    x = x_min + (i + 0.5) * dx
    y = y_min + (j + 0.5) * dy
    z = z_min + (k + 0.5) * dz

    lam_i = find_pi_line_lambda_gpu(x, y, z, R, P, lambda_array, num_angles)
    if lam_i >= 0:
        lam_o = lam_i + math.pi

    value_acc = 0.0
    for idx in range(num_angles):
        lam = lambda_array[idx]
        if lam < lam_i or lam > lam_o:
            continue

        x_s = R * math.cos(lam)
        y_s = R * math.sin(lam)
        z_s = P * lam / (2 * math.pi)

        detector_type = 'flat'
        if detector_type == 'flat':
            v_star = R - x * math.cos(lam) - y * math.sin(lam)
            alpha_star = D/v_star * (-x * math.sin(lam) + y * math.cos(lam))
            w_star = (D / v_star) * (z - z_s)
        else:
            v_star = R - x * math.cos(lam) - y * math.sin(lam)
            alpha_star = math.atan2(-x * math.sin(lam) + y * math.cos(lam), v_star)
            w_star = (D * math.cos(alpha_star) / v_star) * (z - z_s)

        alpha_max = pixel_spacing_u * (det_cols - 1) / 2
        w_max = pixel_spacing_v * (det_rows - 1) / 2

        alpha_idx = (alpha_star + alpha_max) / (2 * alpha_max) * (det_cols - 1)
        w_idx = (w_star + w_max) / (2 * w_max) * (det_rows - 1)

        if 0 <= alpha_idx < det_cols - 1 and 0 <= w_idx < det_rows - 1:
            i0 = int(math.floor(w_idx))
            j0 = int(math.floor(alpha_idx))
            dw = w_idx - i0
            da = alpha_idx - j0

            p00 = input_array[idx, i0, j0]
            p01 = input_array[idx, i0, j0 + 1]
            p10 = input_array[idx, i0 + 1, j0]
            p11 = input_array[idx, i0 + 1, j0 + 1]

            interp = (
                (1 - dw) * (1 - da) * p00 +
                (1 - dw) * da * p01 +
                dw * (1 - da) * p10 +
                dw * da * p11
            )

            value_acc += interp / v_star

    rec_volume[i, j, k] = value_acc / (2 * math.pi)
