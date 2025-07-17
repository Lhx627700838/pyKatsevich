# -----------------------------------------------------------------------
# This file is part of Pykatsevich distribution (https://github.com/astra-toolbox/helical-kats).
# Copyright (c) 2024 imec-Vision Lab, University of Antwerp.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------

"""
This code is based on the implementation from the Cph CT toolbox.
"""

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import cupy as cp
import astra
from time import time 

def differentiate(
        sinogram,
        conf,
        tqdm_bar=False
    ):
    """
    Derivative at constant direction and length correction weighting.
    
    Parameters:
    ===========
    sinogram : ndarray of float
        The 3D array with projection data. The expected shape is (views, det_rows, det_cols).
    conf : dict
        Dictionary with the helical configuration used in the katsevich formula. See initialize.create_configuration() for details.
    tqdm_bar : bool
        Show tqdm progress bar (ignored if tqdm is not installed). The defauls it False.
    
    Returns:
    ========
    output_array : ndarray
        Projection data derivaties.
    """

    try:
        from tqdm import tqdm
        tqdm_imported = True
    except:
        tqdm_imported = False

    output_array = np.zeros_like(sinogram)

    delta_s = conf['delta_s']
    pixel_height = conf['pixel_height']
    pixel_span   = conf['pixel_span']
    dia=conf['scan_diameter']
    dia_sqr = dia ** 2

    col_coords = conf['col_coords'][:-1]
    row_coords = conf['row_coords'][:-1]

    # Helper variables...
    row_col_prod = np.zeros_like(sinogram[0, :-1, :-1])
    row_col_prod += col_coords
    row_transposed = np.zeros_like(row_coords)
    row_transposed += row_coords
    row_transposed.shape = (len(row_coords), 1)
    row_col_prod *= row_transposed
    col_sqr = np.zeros_like(sinogram[0, :-1, :-1])
    col_sqr += col_coords ** 2
    row_sqr = np.zeros_like(sinogram[0, :-1, :-1])
    row_sqr += row_transposed ** 2

    range_object = tqdm(range(0, sinogram.shape[0]-1), "Derivatives   ")

    for proj_index in range_object:
        proj = proj_index - 0 # Since the original code works with data chunks, "- 0" remains here, where "0" is the hardcoded value of the first projection index

        # Differentiation with respect to projections, rows and columns.
        # Expects input to have that order of dimensions!
        # Use the chain rule with neighboring pixels on adjacent projections

        d_proj = (sinogram[proj + 1, :-1, :-1] - sinogram[proj, :
                  -1, :-1] + sinogram[proj + 1, 1:, :-1]
                  - sinogram[proj, 1:, :-1] + sinogram[proj + 1, :
                  -1, 1:] - sinogram[proj, :-1, 1:]
                  + sinogram[proj + 1, 1:, 1:] - sinogram[proj, 1:
                  , 1:]) / (4 * delta_s)
        d_row = (sinogram[proj, 1:, :-1] - sinogram[proj, :-1, :
                 -1] + sinogram[proj, 1:, 1:] - sinogram[proj, :
                 -1, 1:] + sinogram[proj + 1, 1:, :-1]
                 - sinogram[proj + 1, :-1, :-1] + sinogram[proj
                 + 1, 1:, 1:] - sinogram[proj + 1, :-1, 1:]) / (4
                * pixel_height)
        d_col = (sinogram[proj, :-1, 1:] - sinogram[proj, :-1, :
                 -1] + sinogram[proj, 1:, 1:] - sinogram[proj, 1:
                 , :-1] + sinogram[proj + 1, :-1, 1:]
                 - sinogram[proj + 1, :-1, :-1] + sinogram[proj
                 + 1, 1:, 1:] - sinogram[proj + 1, 1:, :-1]) / (4
                * pixel_span)
        output_array[proj, :-1, :-1] = d_proj + d_col * (col_sqr
                + dia_sqr) / dia + d_row * row_col_prod / dia

        # In-place length correction because detector is flat

        output_array[proj, :-1, :-1] *= dia / np.sqrt(col_sqr + dia_sqr + row_sqr)

    return output_array

def fw_Kcurve_rebinning(input_array, conf):
    import numpy as np
    from tqdm import tqdm

    D = conf['scan_diameter'] # SDD
    P = conf['progress_per_turn']
    R0 = conf['scan_radius'] #
    pixel_height = conf['pixel_height']
    detector_columns_coordinate = conf['col_coords']
    detector_rows = conf['detector rows']
    detector_columns = conf['detector cols']
    detector_row_offset = 0
    detector_rebin_rows = conf['detector_rebin_rows']
    

    alpha_m = 0.426235
    M = conf['M']
    psi_list = 1*np.linspace(-np.pi/2 - alpha_m, np.pi/2 + alpha_m, M)
    detector_rebin_rows= M
    output_array = np.zeros((input_array.shape[0], detector_rebin_rows, detector_columns), dtype=np.float32)
    wk_index_map = np.zeros((M, detector_columns), dtype=np.float32)
    angles_range = -1
    if angles_range<0:
        detector_columns_coordinate = 1*detector_columns_coordinate
        psi_list = -1*psi_list
    # ==== 计算 wk_index_map，只执行一次 ====
    tan_psi = np.tan(psi_list)
    tan_psi[np.abs(tan_psi) < 1e-6] = 1e-6  # 避免除0
    term = psi_list / tan_psi


    for col in range(detector_columns):
        u = detector_columns_coordinate[col]
        w_k = -1*(D * P / (2 * np.pi * R0)) * (psi_list + term * (u / D))
        w_k_index = w_k / pixel_height + 0.5 * detector_rows - detector_row_offset
        w_k_index = np.clip(w_k_index, 0.0, detector_rows - 2.001)  # 保证 idx_floor+1 不越界

        wk_index_map[:, col] = w_k_index




    idx_floor = np.floor(wk_index_map).astype(np.int32)  # shape: (detector_rebin_rows, detector_cols)
    frac = wk_index_map - idx_floor

    # ==== 插值执行 ====

    for proj in tqdm(range(input_array.shape[0]), "Forward rebin."):
        data = input_array[proj]  # shape: (detector_rows, detector_cols)
        for col in range(detector_columns):
            floor_idx = idx_floor[:, col]
            f = frac[:, col]
            output_array[proj, :, col] = (1 - f) * data[floor_idx, col] + f * data[floor_idx + 1, col]

    return output_array, wk_index_map, psi_list


def fw_Kcurve_rebinning_fast(input_array, conf, M=47):
    import numpy as np

    # === 读取配置 ===
    D = conf['scan_radius']
    P = conf['progress_per_turn']
    R0 = conf['scan_diameter']
    pixel_height = conf['pixel_height']
    detector_rows = conf['detector rows']
    detector_columns = conf['detector cols']
    detector_row_offset = 0
    detector_rebin_rows = conf['detector_rebin_rows']

    # 确保 col_coords 长度一致
    col_coords = conf['col_coords'][:detector_columns]  # 修正关键点

    # === 构造 psi 和 u 网格 ===
    alpha_m = 0.426235
    psi_list = np.linspace(-np.pi / 2 - alpha_m, np.pi / 2 + alpha_m, 2 * M + 1)  # shape: (95,)
    psi_grid, u_grid = np.meshgrid(psi_list, col_coords, indexing='ij')  # shape: (95, 1376)

    # 避免除 0
    tan_psi = np.tan(psi_grid)
    tan_psi[np.abs(tan_psi) < 1e-6] = 1e-6
    term = psi_grid / tan_psi

    # === 计算 wk 映射 ===
    w_k = (D * P / (2 * np.pi * R0)) * (psi_grid + term * (u_grid / D))
    w_k_index = w_k / pixel_height + 0.5 * detector_rows - detector_row_offset
    w_k_index = np.clip(w_k_index, 0.0, detector_rows - 2.001)

    idx_floor = np.floor(w_k_index).astype(np.int32)  # shape: (95, 1376)
    frac = w_k_index - idx_floor

    # 保存 index map
    wk_index_map = w_k_index.astype(np.float32)

    # === 插值 ===
    n_proj = input_array.shape[0]
    output_array = np.empty((n_proj, detector_rebin_rows, detector_columns), dtype=np.float32)

    col_idx = np.broadcast_to(np.arange(detector_columns), idx_floor.shape)  # shape: (95, 1376)

    for proj in range(n_proj):
        data = input_array[proj]  # shape: (detector_rows, detector_columns)

        # gather data for interpolation
        gather1 = data[idx_floor, col_idx]               # shape: (95, 1376)
        gather2 = data[idx_floor + 1, col_idx]           # shape: (95, 1376)

        # 线性插值
        output_array[proj] = (1 - frac) * gather1 + frac * gather2

    return output_array, wk_index_map, psi_list



def plot_kappa_curves(wk_index_map, psi_list, detector_columns_coordinate, pixel_height):
    w_axis = (wk_index_map - 72) * pixel_height  # 将 index 转为物理坐标（cm）
    alpha = detector_columns_coordinate  # 横轴

    plt.figure(figsize=(8, 6))

    # 逐行绘制每一条 κ-curve
    for i in range(wk_index_map.shape[0]):
        plt.plot(alpha, w_axis[i, :], 'k')  # 黑色线条

    # 画出 detector row 的中心位置
    for j in range(144):
        w_j = (j - 144 / 2) * pixel_height
        plt.axhline(w_j, color='red', linestyle=':', linewidth=0.5)

    # 可视化标注
    plt.xlabel(r'$\alpha$ (in radians)')
    plt.ylabel(r'$w$ (cm)')
    plt.title(r'$\kappa$-curves for rebinning')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_hilbert_kernel_new(conf):
    kernel_radius = conf['kernel_radius']
    N = 2 * kernel_radius + 1  # total length
    proj_filter_array = np.zeros(N, dtype=np.float32)

    for i in range(N):
        n = i - kernel_radius
        if n == 0:
            proj_filter_array[i] = 0.0  # h(0) = 0
        else:
            proj_filter_array[i] = 1.0 / (np.pi * n)

    return proj_filter_array


def compute_hilbert_kernel(
        conf
    ):
    from tqdm import tqdm
    proj_filter_array = np.zeros(conf['kernel_width'], dtype=np.float32)

    # We use a simplified hilbert kernel for now

    kernel_radius = conf['kernel_radius']
    for i in tqdm(range(conf['kernel_width'])):
        proj_filter_array[i] = (1.0 - np.cos(np.pi * (i - kernel_radius - 0.5))) \
                            / (np.pi * (i - kernel_radius - 0.5))
    
    return proj_filter_array



def hilbert_conv_ff(input_array, hilbert_array, conf):
    from scipy.signal import fftconvolve
    import numpy as np
    from tqdm import tqdm
    detector_columns = conf['detector cols']
    detector_rebin_rows = conf['M']

    output_array = np.zeros_like(input_array)
    print(np.shape(input_array))
    print(detector_rebin_rows)
    for proj in tqdm(range(input_array.shape[0])):
        for rebin_row in range(detector_rebin_rows):
            # 使用 fftconvolve 替代普通卷积
            filtered = fftconvolve(
                input_array[proj, rebin_row, :],
                hilbert_array,
                mode='full'
            )
            # 取中心部分
            output_array[proj, rebin_row, :] = filtered[detector_columns - 1:2 * detector_columns - 1]

    return output_array

def hilbert_conv_gpu(input_array, hilbert_array, conf):
    import cupy as cp
    from cupy.fft import fft, ifft
    from tqdm import tqdm

    # 输入数据 shape
    n_proj, n_rebin_rows, n_cols = input_array.shape
    L_filter = hilbert_array.shape[0]
    n_fft = n_cols + L_filter - 1

    # 上传 Hilbert kernel 到 GPU，并执行 FFT
    hilbert_gpu = cp.asarray(hilbert_array, dtype=cp.float32)
    hilbert_fft = fft(hilbert_gpu, n=n_fft)

    # 上传输入数据到 GPU
    input_gpu = cp.asarray(input_array, dtype=cp.float32)

    # 创建输出数组
    output_gpu = cp.zeros_like(input_gpu)

    for proj in tqdm(range(n_proj), desc="Hilbert GPU FFT"):
        # shape: (rebin_rows, cols)
        data = input_gpu[proj]
        data_fft = fft(data, n=n_fft, axis=1)  # shape: (rows, n_fft)
        result_fft = data_fft * hilbert_fft[cp.newaxis, :]  # broadcasting
        result = cp.real(ifft(result_fft, axis=1))[:, (L_filter - 1):(L_filter - 1 + n_cols)]
        output_gpu[proj] = result

    # 下载回 CPU
    output_array = cp.asnumpy(output_gpu)

    return output_array

def hilbert_conv(
        input_array,
        hilbert_array,
        conf
    ):
    from tqdm import tqdm
    from numpy import convolve

    detector_columns = conf['detector cols']
    detector_rebin_rows = conf['detector_rebin_rows']

    output_array = np.zeros_like(input_array)
    # TODO: use rectangular hilbert window as suggested in Noo paper?

    for proj_index in tqdm(range(0, input_array.shape[0])):
        proj = proj_index - 0
        for rebin_row in range(detector_rebin_rows):

            # use convolve for now instead of manual convolution sum
            # yields len(hilbert_array)+detector_columns-1 elements

            filter_conv = convolve(hilbert_array, input_array[proj,
                                   rebin_row, :])

            # only use central elements of convolution

            tmp = filter_conv[detector_columns - 1:2 * detector_columns
                - 1]
            output_array[proj, rebin_row, :] = tmp
    
    return output_array
def hilbert_conv_gpu_batched(input_array, hilbert_array, conf, batch_size=200, gpu_id=0):
    import cupy as cp
    from cupy.fft import fft, ifft
    from tqdm import tqdm
    import numpy as np

    n_proj, n_rebin_rows, n_cols = input_array.shape
    L_filter = hilbert_array.shape[0]
    n_fft = n_cols + L_filter - 1  # for full convolution

    output_array = np.zeros_like(input_array, dtype=np.float32)

    with cp.cuda.Device(gpu_id):  # 安全地绑定 GPU
        hilbert_gpu = cp.asarray(hilbert_array, dtype=cp.float32)
        hilbert_fft = fft(hilbert_gpu, n=n_fft)

        for start in tqdm(range(0, n_proj, batch_size), desc="Hilbert GPU FFT Batches"):
            end = min(start + batch_size, n_proj)
            B = end - start

            # shape (B, H, W)
            batch_cpu = input_array[start:end]
            batch_gpu = cp.asarray(batch_cpu, dtype=cp.float32)

            # padding to match full convolution
            pad_width = n_fft - n_cols
            batch_gpu_padded = cp.pad(batch_gpu, ((0, 0), (0, 0), (0, pad_width)), mode='constant')

            # FFT along last axis
            batch_fft = fft(batch_gpu_padded, axis=2)
            result_fft = batch_fft * hilbert_fft[cp.newaxis, cp.newaxis, :]
            result_ifft = ifft(result_fft, axis=2)

            # crop: center part = convolution output (same as np.convolve(..., mode='full')) center
            result = cp.real(result_ifft[:, :, (L_filter - 1):(L_filter - 1 + n_cols)])

            # bring back to CPU
            output_array[start:end] = cp.asnumpy(result)

    return output_array


def hilbert_trans_scipy(input_array):
    from scipy.signal import hilbert
    output_array = np.imag(hilbert(input_array, axis=2))
    return output_array

def bw_Kcurve_rebinning(input_array, wk_index_map, conf):
    import numpy as np
    from tqdm import tqdm

    detector_rows = conf['detector rows']
    detector_columns = conf['detector cols']
    detector_row_offset = 0

    n_proj = input_array.shape[0]
    n_psi = wk_index_map.shape[0]

    output_array = np.zeros((n_proj, detector_rows, detector_columns), dtype=np.float32)

    for proj in tqdm(range(n_proj), "Backward rebin."):
        for col in range(detector_columns):
            for row in range(detector_rows):
                w_idx = row - 0.5 * detector_rows + detector_row_offset  # physical w location

                # 找到 ψ_idx 满足 wk_index_map[ψ_idx, col] 最接近 w_idx
                wk_col = wk_index_map[:, col]
                if wk_col[0] > w_idx or wk_col[-1] < w_idx:
                    continue  # 超出范围，不插值

                idx_upper = np.searchsorted(wk_col, w_idx, side='right')
                idx_lower = idx_upper - 1

                if idx_upper >= n_psi or idx_lower < 0:
                    continue

                wk0 = wk_col[idx_lower]
                wk1 = wk_col[idx_upper]
                f0 = input_array[proj, idx_lower, col]
                f1 = input_array[proj, idx_upper, col]

                # 线性插值
                if wk1 == wk0:
                    val = f0
                else:
                    weight = (w_idx - wk0) / (wk1 - wk0)
                    val = (1 - weight) * f0 + weight * f1

                output_array[proj, row, col] = val

    return output_array

def bw_Kcurve_rebinning_fast(input_array, wk_index_map, conf ,oringinaldata):
    import numpy as np
    from tqdm import tqdm

    detector_rows = conf['detector rows']
    detector_columns = conf['detector cols']
    detector_row_offset = 0

    n_proj = input_array.shape[0]
    n_psi = wk_index_map.shape[0]

    output_array = np.zeros((n_proj, detector_rows, detector_columns), dtype=np.float32)
    # output_array = oringinaldata
    # 对每个列做一次插值映射（因为 wk_index_map 只跟列相关）
    for col in tqdm(range(detector_columns), desc="Backward rebin."):
        wk_col = wk_index_map[:, col]

        # 提前预取该列所有插值坐标及权重
        for row in range(detector_rows):
            w_idx = row

            # 若当前 row 对应的 w 不在 wk_col 的范围内，则跳过
            if wk_col[0] > w_idx or wk_col[-1] < w_idx:
                continue

            # 查找插值点上下界索引
            idx_upper = np.searchsorted(wk_col, w_idx, side='right')
            idx_lower = idx_upper - 1

            if idx_upper >= n_psi or idx_lower < 0:
                continue  # 越界则跳过

            wk0 = wk_col[idx_lower]
            wk1 = wk_col[idx_upper]
            w = (w_idx - wk0) / (wk1 - wk0) if wk1 != wk0 else 0.0

            # 批量插值：对所有投影进行线性插值
            f0 = input_array[:, idx_lower, col]  # shape: (n_proj,)
            f1 = input_array[:, idx_upper, col]  # shape: (n_proj,)
            val = (1 - w) * f0 + w * f1

            # 写回所有投影
            output_array[:, row, col] = val

    return output_array



def filter_katsevich(
    input_array: np.ndarray, # The shape expected is (views, rows, columns)
    conf: dict,
    verbosity_options: dict = {}
):
    """
    Run all filtering steps.

    Parameters:
    ===========
    input_array : ndarray
        Input array. The shape expected is (proejction views, rows, columns).
    conf : dict
        Configuration dictionary.
    verbosity_options : dict
        Dictionary with verbosity options for each step.
        Steps have keys ["Diff", "FwdRebin", "BackRebin"].
        Each filtering step key corresponds to another dictionary with keys ["Progress bar", "Print time"].

    """
    differentite_opts = verbosity_options.get("Diff", {})
    diff_tqdm_bar = differentite_opts.get("Progress bar", False)
    diff_print_time = differentite_opts.get("Print time", False)

    if diff_print_time and not diff_tqdm_bar:
        print("Derivative at constant direction step", end="... ")
    t1 = time()
    
    diff_proj = differentiate(input_array, conf, diff_tqdm_bar)
    t2 = time()
    if diff_print_time:
        print(f"differentiate Done in {t2-t1:.4f} seconds")

    fwd_rebin_opts = verbosity_options.get("FwdRebin", {})
    fwd_rebin_tqdm_bar = fwd_rebin_opts.get("Progress bar", False)
    fwd_rebin_time = fwd_rebin_opts.get("Print time", False)

    if fwd_rebin_time and not fwd_rebin_tqdm_bar:
        print("Forward height rebinning step", end="... ")
    t1 = time()
    #fwd_rebin_array = fw_height_rebinning_fast(diff_proj, conf, fwd_rebin_tqdm_bar)
    fwd_rebin_array, wk_index_map, psi_list = fw_Kcurve_rebinning(diff_proj, conf)
    plot_kappa_curves(wk_index_map, psi_list, conf['col_coords'], conf['pixel_height'])
    t2 = time()
    if fwd_rebin_time:
        print(f"fhr Done in {t2-t1:.4f} seconds")
    
    hilbert_array = compute_hilbert_kernel_new(conf)
    sino_hilbert_trans = hilbert_conv_ff(fwd_rebin_array,hilbert_array,conf)
    #sino_hilbert_trans_gpu = hilbert_conv_gpu_batched(fwd_rebin_array,hilbert_array,conf,batch_size=200,gpu_id=0)
    #print("Max diff:", np.max(np.abs(sino_hilbert_trans - sino_hilbert_trans_gpu)))
    cp.get_default_memory_pool().free_all_blocks()
    print('hilbert done')
    back_rebin_opts = verbosity_options.get("BackRebin", {})
    back_rebin_tqdm_bar = back_rebin_opts.get("Progress bar", False)
    back_rebin_time = back_rebin_opts.get("Print time", False)

    if back_rebin_time and not back_rebin_tqdm_bar:
        print("Backward height rebinning step", end="... ")
    t1 = time()
    #filtered_projections = rev_rebin_vec_fast(sino_hilbert_trans_gpu, conf, back_rebin_tqdm_bar)
    filtered_projections = bw_Kcurve_rebinning_fast(sino_hilbert_trans, wk_index_map, conf,input_array)
    t2 = time()
    if back_rebin_time:
        print(f"bhr Done in {t2-t1:.4f} seconds")

    saveit = 1
    if saveit == 1:    
        import tifffile
        tifffile.imwrite('filtered_proj1.tif', input_array.astype(np.float32))
        tifffile.imwrite('filtered_proj2.tif', diff_proj.astype(np.float32))
        tifffile.imwrite('filtered_proj3.tif', fwd_rebin_array.astype(np.float32))
        tifffile.imwrite('filtered_proj4.tif', sino_hilbert_trans.astype(np.float32))
        tifffile.imwrite('filtered_proj5.tif', filtered_projections.astype(np.float32))
    return filtered_projections

def sino_weight_td(input_array, conf, show_td_window=True):
    w_bottom = np.reshape(conf['proj_row_mins'], (1, -1))
    w_top    = np.reshape(conf['proj_row_maxs'], (1, -1))

    dw = conf['pixel_height']
    a  = conf['T-D smoothing']
    W, U = np.meshgrid(conf['row_coords'], conf['col_coords'], indexing='ij')
    print(np.shape(W))
    # print(f'W = {U}')

    def chi(a, U_mesh, W_mesh):
        mask = np.zeros_like(W_mesh)

        # 提前展开上下界区间
        W_top_high     = np.repeat(w_top + a*dw, W_mesh.shape[0], axis=0)
        W_top_low      = np.repeat(w_top - a*dw, W_mesh.shape[0], axis=0)
        W_bottom_high  = np.repeat(w_bottom + a*dw, W_mesh.shape[0], axis=0)
        W_bottom_low   = np.repeat(w_bottom - a*dw, W_mesh.shape[0], axis=0)

        # 中心区域：设为1
        mask[(W_mesh > W_bottom_high) & (W_mesh < W_top_low)] = 1

        # 上过渡区域
        idx_upper = (W_mesh > W_top_low) & (W_mesh < W_top_high)
        mask[idx_upper] = (W_top_high[idx_upper] - W_mesh[idx_upper]) / (2 * a * dw)

        # 下过渡区域
        idx_lower = (W_mesh > W_bottom_low) & (W_mesh < W_bottom_high)
        mask[idx_lower] = (W_mesh[idx_lower] - W_bottom_low[idx_lower]) / (2 * a * dw)

        return mask

    
    TD_mask = chi(a, U, W)
    if show_td_window:
        #plt.figure()
        #plt.imshow(TD_mask, cmap='gray')
        #plt.colorbar()
        import tifffile
        tifffile.imwrite('TD_mask.tif',TD_mask[:,::-1])
        # plt.show()

    sino_td_weighted = np.zeros_like(input_array)
    for proj in range(input_array.shape[0]):
        sino_td_weighted[proj] = input_array[proj] * TD_mask

    return sino_td_weighted

def backproject_a(
        input_array,
        conf,
        vol_geom,
        proj_geom,
        tqdm_bar=False
    ):
    """
    Run Katsevich's backprojection using ASTRA's BP3D_CUDA algorithm.
    Note that the CuPy memory pool will not be cleaned even after the function return.
    See https://docs.cupy.dev/en/stable/user_guide/memory.html for details.

    Parameters
    ----------
    input_array : ndarray
        Filtered projections. Expected shape is (angles, rows, columns).
    conf : dict
        Configuration dictionary.
    vol_geom : dict
        ASTRA volume geometry dictionary.
    proj_geom : dict
        ASTRA projection geometry dictionary.
    tqdm_bar : bool
        Show tqdm progress bar (ignored if tqdm is not installed). The defauls it False.
    Returns
    -------
    output : ndarray
        Reconstructed 3D XCT image.
    """

    try:
        from tqdm import tqdm
        tqdm_imported = True
    except:
        tqdm_imported = False

    x_voxels = conf['x_voxels']
    y_voxels = conf['y_voxels']
    x_min = conf['x_min']
    y_min = conf['y_min']
    delta_x = conf['delta_x']
    delta_y = conf['delta_y']

    scan_radius = conf['scan_radius']
    source_pos = conf['source_pos']

    scale_integrate_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void scale_integrate_cu(float* bp_volume, float *rec_volume, float xsize, float xmin, float ysize, float ymin, float angle, float scan_radius, float scale_const, uint3 vol_dims) {
        
        unsigned int gridRowInd = blockDim.x * blockIdx.x + threadIdx.x; 
        unsigned int gridColInd = blockDim.y * blockIdx.y + threadIdx.y;
        unsigned int gridSliInd = blockDim.z * blockIdx.z + threadIdx.z;
                                
        unsigned int gridRowStride = blockDim.x * gridDim.x;
        unsigned int gridColStride = blockDim.y * gridDim.y;
        unsigned int gridSliStride = blockDim.z * gridDim.z;
        
        float ang_sin, ang_cos;
        sincosf(angle, &ang_sin, &ang_cos);
        
        for(unsigned int j_col = gridColInd; j_col < vol_dims.y; j_col += gridColStride)
        {
            for(unsigned int k_sli = gridSliInd; k_sli < vol_dims.z; k_sli += gridSliStride)
            {
                // The volume shape - x for rows, y for columns, z for slices, is shaped so that axes follow as (z, y, x)

                float X = xsize*k_sli + xmin + 0.5f*xsize;
                float Y = ysize*j_col + ymin + 0.5f*ysize;

                float scale_coeff = -1*scale_const*(scan_radius - X*ang_cos - Y*ang_sin);
                    
                for(unsigned int i_row = gridRowInd; i_row < vol_dims.x; i_row += gridRowStride)
                {
                    unsigned long long tid = k_sli + j_col*vol_dims.z + i_row * vol_dims.z * vol_dims.y;
                    rec_volume[tid] += bp_volume[tid] / scale_coeff;
                }
            }
        }
    }
    ''', 'scale_integrate_cu')

    sino_for_astra = np.asarray(np.swapaxes(input_array, 1, 0), dtype=np.float32, order='C')

    algorithm_name = "BP3D_CUDA"
    cfg = astra.astra_dict(algorithm_name)

    pixel_size = conf['pixel_span']

    sod = conf['scan_radius']
    sdd = conf['scan_diameter']

    # Sinogram on GPU as a CuPy array:
    sino_astra_cp = cp.array(sino_for_astra, dtype=cp.float32, blocking=True)

    # CuPy arrays + ASTRA GPULink
    # Reconstruction volume:
    rec_volume_cp = cp.zeros(astra.geom_size(vol_geom), dtype=cp.float32)

    # Backprojection volume:
    bp_astra_cp = cp.zeros(astra.geom_size(vol_geom), dtype=cp.float32)

    z, y, x = bp_astra_cp.shape
    bp_astra_link = astra.data3d.GPULink(bp_astra_cp.data.ptr, x, y, z, bp_astra_cp.strides[-2])
    bp_astra_id = astra.data3d.link('-vol', vol_geom, bp_astra_link)

    # CUDA kernel grid:
    blocksize_z: int = min(bp_astra_cp.shape[2], 64)
    blocksize_y: int = min(bp_astra_cp.shape[1], 4)
    blocksize_x: int = min(bp_astra_cp.shape[0], 2)

    num_blocks = lambda n_elems, blocksize : (n_elems + blocksize - 1) // blocksize

    numBlocks_x: int = min( num_blocks(bp_astra_cp.shape[0], blocksize_x), 2**31 - 1 )
    numBlocks_y: int = min( num_blocks(bp_astra_cp.shape[1], blocksize_y), 2**31 - 1 )
    numBlocks_z: int = min( num_blocks(bp_astra_cp.shape[2], blocksize_z), 65535 )
    print(f'Block size: {blocksize_x, blocksize_y, blocksize_z}, num blocks: {numBlocks_x, numBlocks_y, numBlocks_z}')

    # Dimensions struct to be passed to the CUDA kernel:
    dims3 = np.dtype({'names': ["rows", "columns", "slices"], 'formats': [np.uint32]*3})
    bp_dims3 = np.asarray(bp_astra_cp.shape, dtype=np.uint32).view(dims3)

    # Assuming cubic voxel and contant magnification for all voxel pisitions:
    astra_bp_scaling = (delta_x**3) / ( ( pixel_size/ (sdd / sod) )**2 )
    # Add integration step to scaling:
    scale_coeff = astra_bp_scaling * conf['projs_per_turn']
    range_object = tqdm(range(proj_geom['Vectors'].shape[0]), "Backprojection (Kernel)") if tqdm_imported and tqdm_bar else range(proj_geom['Vectors'].shape[0])

    for proj_angle in range_object:

        proj_geom_view = astra.create_proj_geom(
            "cone_vec",
            proj_geom["DetectorRowCount"],
            proj_geom["DetectorColCount"],
            np.reshape(proj_geom["Vectors"][proj_angle], (1, -1)) # 1x12
        )
        
        # Projection data as CuPy + ASTRA GPULink:
        proj_astra_cp = cp.expand_dims(sino_astra_cp[:, proj_angle, :], 1)
        z, y, x = proj_astra_cp.shape
        sino_astra_link = astra.data3d.GPULink(proj_astra_cp.data.ptr, x,y,z, proj_astra_cp.strides[-2])

        sino_astra_id = astra.data3d.link(
            '-sino',
            proj_geom_view,
            sino_astra_link
        )

        cfg['ReconstructionDataId'] = bp_astra_id
        cfg['ProjectionDataId'] = sino_astra_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        astra.algorithm.delete([alg_id])
        astra.data3d.delete([sino_astra_id]) # Only remove astra objects, memory is not deallocated

        angle = source_pos[proj_angle]

        scale_integrate_kernel(
            (numBlocks_x, numBlocks_y, numBlocks_z),
            (blocksize_x, blocksize_y, blocksize_z),
            (
                bp_astra_cp,
                rec_volume_cp,
                cp.float32(delta_x),
                cp.float32(x_min),
                cp.float32(delta_y),
                cp.float32(y_min),
                cp.float32(angle),
                cp.float32(scan_radius),
                cp.float32(scale_coeff),
                bp_dims3
            )
        )
        
    astra.data3d.delete([bp_astra_id])

    rec_volume = np.asarray(np.moveaxis(rec_volume_cp.get(), 0, 2), order='C')

    return rec_volume