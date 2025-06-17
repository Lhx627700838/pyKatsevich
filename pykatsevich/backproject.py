'''
“滤波后的投影数据 → 每个角度做 cone-beam 反投影 → 用 Katsevich 权重因子积分 → 得到最终重建图像”

✅ 功能简介
该函数会根据：

给定体素位置 (x, y, z)

已知的螺旋轨道参数 R（扫描半径）和 P（螺距）

所有 λ（投影角）

'''
import numpy as np
def find_pi_line_range(x, y, z, lambdas, R, P):
    lam_i = find_pi_line_through_point(x, y, z, R, P, lambdas)
    lam_o = lam_i + np.pi
    # Clip to available λ range
    lam_i = max(lambdas[0], lam_i)
    lam_o = min(lambdas[-1], lam_o)
    # 返回索引范围
    valid_indices = np.where((lambdas >= lam_i) & (lambdas <= lam_o))[0]
    return valid_indices

DEBUG = False
def find_pi_line_through_point(x, y, z, R, P, lambda_array):
    best_lambda = None
    min_error = 1e10
    print("finding pi line for:", x, y ,z)
    count = 0
    for lam_i in lambda_array:
        lam_o = lam_i + np.pi
        if lam_o > lambda_array[-1]:
            break
        
        ai = np.array([R*np.cos(lam_i), R*np.sin(lam_i), P*lam_i/(2*np.pi)])
        ao = np.array([R*np.cos(lam_o), R*np.sin(lam_o), P*lam_o/(2*np.pi)])
        
        v = ao - ai
        p = np.array([x, y, z])

        t = np.dot(p - ai, v) / np.dot(v, v)
        if 0.0 <= t <= 1.0:
            proj = ai + t * v
            dist = np.linalg.norm(proj - p)
            if dist < min_error:
                min_error = dist
                best_lambda = lam_i
                best_ai = ai
        count += 1
        if DEBUG and count%10==0:
            print("check line: ")
            print("lam_i, ai - lam_o, ao:", lam_i, ai, lam_o, ao)
            print("t for line:", t)
            print("distance between line and point:", dist)
    print("best_lambda lam_i, ai: ", best_lambda, best_ai)
    print("min error: ", min_error)
    return best_lambda #if min_error < 10.0 else None  # 用 1mm 容差限制

def katsevich_backprojection_cpu(
    input_array,
    conf,
    volume_shape,
    lambda_array,
    tqdm_bar=True
):
    """
    Perform Katsevich backprojection with π-line restriction (exact formula).
    Only for educational / validation purposes due to high cost.

    Parameters
    ----------
    input_array : ndarray, shape (num_angles, det_rows, det_cols)
        The filtered projection data g_F(λ, α, w).
    conf : dict
        Configuration dict with keys: delta_x, delta_y, delta_z, x_min, y_min, z_min, R, P.
    volume_shape : tuple (nx, ny, nz)
        Shape of the reconstruction volume.
    lambda_array : ndarray
        Array of projection angles (λ), in radians.
    tqdm_bar : bool
        Show progress bar.

    Returns
    -------
    rec_volume : ndarray
        Reconstructed volume.
    """
    from tqdm import tqdm

    nx, ny, nz = volume_shape
    dx, dy, dz = conf['delta_x'], conf['delta_y'], conf['delta_z']
    x_min, y_min, z_min = conf['x_min'], conf['y_min'], conf['z_min']
    R = conf['scan_radius']
    P = conf['progress_per_turn']

    D = conf['scan_diameter']
    pixel_spacing_u = conf['detector_pixel_span_u']
    pixel_spacing_v = conf['detector_pixel_span_v']

    rec_volume = np.zeros((nx, ny, nz), dtype=np.float32)
    num_angles = len(lambda_array)

    iterator = tqdm(np.ndindex(nx, ny, nz), desc="Katsevich Exact BP") if tqdm_bar else np.ndindex(nx, ny, nz)

    for i, j, k in iterator:
        x = x_min + (i + 0.5) * dx
        y = y_min + (j + 0.5) * dy
        z = z_min + (k + 0.5) * dz

        # Find valid π-line support range
        valid_indices = find_pi_line_range(x, y, z, lambda_array, R, P)

        for idx in valid_indices:
            lam = lambda_array[idx]

            # Source position at angle λ
            x_s = R * np.cos(lam)
            y_s = R * np.sin(lam)
            z_s = P * lam / (2 * np.pi)

            # v*(λ, x)
            v_star = R - x * np.cos(lam) - y * np.sin(lam)

            # α*(λ, x)
            alpha_star = np.arctan((-x * np.sin(lam) + y * np.cos(lam)) / v_star)

            # w*(λ, x)
            w_star = (D * np.cos(alpha_star) / v_star) * (z - z_s)

            # Map α*, w* to detector indices
            det_rows, det_cols = input_array.shape[1:]
            alpha_max = pixel_spacing_u * (det_cols - 1) / 2
            w_max = pixel_spacing_v * (det_rows - 1) / 2

            alpha_idx = (alpha_star + alpha_max) / (2 * alpha_max) * (det_cols - 1)
            w_idx = (w_star + w_max) / (2 * w_max) * (det_rows - 1)

            if 0 <= alpha_idx < det_cols - 1 and 0 <= w_idx < det_rows - 1:
                i0, j0 = int(np.floor(w_idx)), int(np.floor(alpha_idx))
                dw, da = w_idx - i0, alpha_idx - j0

                # Bilinear interpolation
                patch = input_array[idx, i0:i0+2, j0:j0+2]
                value = (
                    (1 - dw) * (1 - da) * patch[0, 0] +
                    (1 - dw) * da       * patch[0, 1] +
                    dw       * (1 - da) * patch[1, 0] +
                    dw       * da       * patch[1, 1]
                )

                rec_volume[i, j, k] += value / v_star

        # Final scaling
        rec_volume[i, j, k] *= 1 / (2 * np.pi)

    return rec_volume

