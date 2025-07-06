import cupy as cp
from tqdm import tqdm
#import pudb; pudb.set_trace()
import pdb

import cupy as cp
from tqdm import tqdm

def katsevich_backprojection_curved_gpu1(
    g_filtered, lambdas, x_grid, y_grid, z_grid, R0, D, P, d_alpha, d_w, lambda0=0, z0=0
    
):
    """
    pointwise GPU version:
    - loops over x, y, z
    - each voxel accumulates over all lambdas
    - cupy kernel
    """

    # cupy array
    g_filtered = cp.asarray(g_filtered)
    lambdas = cp.asarray(lambdas)

    Nx, Ny, Nz = x_grid.shape

    x_grid = cp.asarray(x_grid)
    y_grid = cp.asarray(y_grid)
    z_grid = cp.asarray(z_grid)

    d_lambda = cp.abs(lambdas[1] - lambdas[0])
    N_lambda = len(lambdas)

    N_w     = g_filtered.shape[1]
    N_alpha = g_filtered.shape[2]
    

    f = cp.zeros((Nx, Ny, Nz), dtype=cp.float32)

    for ix in tqdm(range(Nx), desc="x loop"):
        for iy in range(Ny):
            for iz in range(Nz):
                x = x_grid[ix, iy, iz]
                y = y_grid[ix, iy, iz]
                z = z_grid[ix, iy, iz]
                accum = 0.0

                for i in range(N_lambda):
                    lam = lambdas[i]
                    phi = lam + lambda0
                    cos_phi = cp.cos(phi)
                    sin_phi = cp.sin(phi)

                    v_star = R0 - x * cos_phi - y * sin_phi
                    #print(v_star)
                    if v_star == 0:
                        continue

                    alpha_star = cp.arctan( (-x*sin_phi + y*cos_phi) / v_star )
                    w_star = D * cp.cos(alpha_star) / v_star * (z - z0 - P * phi / (2*cp.pi))

                    alpha_idx = int(alpha_star/d_alpha + N_alpha/2)
                    w_idx     = int(-w_star/d_w + N_w/2)
                    #pdb.set_trace()
                    if (
                        0 <= alpha_idx < N_alpha
                        and 0 <= w_idx    < N_w
                    ):
                        g_val = g_filtered[i, w_idx, alpha_idx]
                    else:
                        g_val = 0.0

                    accum += (g_val / v_star) * d_lambda

                f[ix, iy, iz] = accum / (2*cp.pi)

    return cp.asnumpy(f)

import numpy as np
def katsevich_backprojection_curved_gpu(
    g_filtered, conf, lambda0=0, z0=0
):  
    # 设置体素空间大小与分辨率
    # Nx, Ny, Nz = 128, 128, 128         # 体素数
    # dx, dy, dz = 4.0, 4.0, 1.0         # 每个体素的尺寸（单位 mm）

    Nx, Ny, Nz = 512, 512, 256         # 体素数
    dx, dy, dz = 0.5, 0.5, 0.5         # 每个体素的尺寸（单位 mm）

    # 定义中心坐标位置 z范围：48-80 * 40
    x_center, y_center, z_center = 0.0, 0.0, 0.0

    # 创建物理坐标范围
    x = np.linspace(-(Nx // 2) * dx + x_center, (Nx // 2 - 1) * dx + x_center, Nx)
    y = np.linspace(-(Ny // 2) * dy + y_center, (Ny // 2 - 1) * dy + y_center, Ny)
    z = np.linspace(-(Nz // 2) * dz + z_center, (Nz // 2 - 1) * dz + z_center, Nz)

    # 构建 3D 网格
    x_grid_fake, y_grid_fake, z_grid = np.meshgrid(x, y, z, indexing='ij')  # shape: (Nx, Ny, Nz)

    x_grid = y_grid_fake
    y_grid = -1 * x_grid_fake
    
    lambdas = conf['source_pos']
    R0 = conf['scan_radius']
    D = conf['scan_diameter'] 
    P = conf['progress_per_turn']
    d_alpha = conf['detector_pixel_span_u']
    d_w = conf['detector_pixel_span_v']

    print('projs_per_turn', conf['projs_per_turn'])

    # 将数据转为 GPU 张量
    g_filtered = cp.asarray(g_filtered)
    lambdas = cp.asarray(lambdas)
    x_grid = cp.asarray(x_grid)
    y_grid = cp.asarray(y_grid)
    z_grid = cp.asarray(z_grid)
    
    #d_alpha = cp.asarray(d_alpha)

    f = cp.zeros_like(x_grid)
    d_lambda = cp.abs(lambdas[1] - lambdas[0])
    N_lambda = len(lambdas)

    N_w = g_filtered.shape[1]
    N_alpha = g_filtered.shape[2]
    
    alpha_max = cp.pi / 2
    w_max = D

    print("g_filtered.shape", g_filtered.shape)
    print('d_alpha:', d_alpha)
    print('P:', P)
    #P = -1 * P
    print('z_grid')
    print(z_grid)

    print('lambdas')
    print(lambdas)

    # should take 0 for point out of border
    # z direction should be checked
    for i in tqdm(range(N_lambda), desc="GPU backprojection"):
        lam = lambdas[i]
        
        phi = lam + lambda0
        # phi *= -1 # lambda increases counterclockwise so should multiply -1
        cos_phi = cp.cos(phi)
        sin_phi = cp.sin(phi)

        v_star = R0 - x_grid * cos_phi - y_grid * sin_phi
        alpha_star = cp.arctan((1.0 / v_star) * (-x_grid * sin_phi + y_grid * cos_phi))
        w_star = D * cp.cos(alpha_star) / v_star * (z_grid - z0 - P * phi / (2 * cp.pi))
        
        alpha_idx = (alpha_star/d_alpha + N_alpha/2).astype(cp.int32)
        w_idx = (-1*w_star/d_w + N_w/2).astype(cp.int32)

        # pdb.set_trace() # p x_grid[:,:,0] p y_grid[:,:,0]
        # 创建合法索引的掩码 
        valid_mask = (
            (alpha_idx >= 0) & (alpha_idx < N_alpha) &
            (w_idx >= 0) & (w_idx < N_w)
        )

        # 初始化采样值为0
        g_sample = cp.zeros_like(v_star)

        # 对合法位置采样
        g_sample[valid_mask] = g_filtered[i, w_idx[valid_mask], alpha_idx[valid_mask]]

        # 加权积分
        f += (g_sample / v_star) * d_lambda

        #print("v_star.shape, alpha_star.shape, w_star.shape",v_star.shape, alpha_star.shape, w_star.shape)
        #print('valid_mask.shape', valid_mask.shape)
        #print('f.shape', f.shape)

    return cp.asnumpy(f / (2 * cp.pi))
