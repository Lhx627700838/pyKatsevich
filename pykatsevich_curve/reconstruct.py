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

import numpy as np

from pykatsevich_curve.filter import filter_katsevich, sino_weight_td, backproject_a
from time import time

def reconstruct(
    input_array,
    conf,
    vol_geom,
    proj_geom,
    verbosity_options: dict = {},
    clear_cupy_mempool=True
):
    """
    Run Katsevich reconsturction with the GPU backprojection.

    Parameters:
    ===========
    input_array : ndarray
        Log-corrected projections. Expected shape is (angles, rows, columns).
    conf : dict
        Configuration dictionary.
    vol_geom : dict
        ASTRA volume geometry dictionary.
    proj_geom : dict
        ASTRA projection geometry dictionary.
    verbosity_options : dict
        Verbosity options for each step of reconstruction. The default is an empty dictionary.
    clear_cupy_mempool : bool
        Clear CuPy default memory and pinned memory pools. The default is True.
    """
    filt_opts = {}
    for k in ("Diff", "FwdRebin", "BackRebin"):
        filt_opts[k] = verbosity_options.get(k, {})
    
    filtered_projections = filter_katsevich(
            input_array,
            conf,
            filt_opts
        )
    
    sino_td = sino_weight_td(filtered_projections, conf, True)
    import tifffile
    tifffile.imwrite('filtered_proj6.tif',sino_td)
    backproject_opts = verbosity_options.get("BackProj", {})
    bp_tqdm_bar = backproject_opts.get("Progress bar", False)
    bp_print_time = backproject_opts.get("Print time", False)

    if bp_print_time and not bp_tqdm_bar:
        print("Backprojection step", end="... ")

    '''t1 = time()
    bp_astra = backproject_a(
        sino_td,
        conf,
        vol_geom,
        proj_geom,
        tqdm_bar=bp_tqdm_bar
    )
    t2 = time()
    if bp_print_time:
        print(f"Done in {t2-t1:.4f} seconds")

    if clear_cupy_mempool:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()'''
    from pykatsevich_curve.backprojection_gpu import katsevich_backprojection_curved_gpu
    import numpy as np

    # 设置体素空间大小与分辨率
    # Nx, Ny, Nz = 128, 128, 128         # 体素数
    # dx, dy, dz = 4.0, 4.0, 1.0         # 每个体素的尺寸（单位 mm）

    Nx, Ny, Nz = 512, 512, 128         # 体素数
    dx, dy, dz = 1.0, 1.0, 1.0         # 每个体素的尺寸（单位 mm）

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

    print(conf['projs_per_turn'])
    bp_astra = katsevich_backprojection_curved_gpu(sino_td, lambdas, x_grid, y_grid, z_grid, R0, D, P, d_alpha, d_w)
    t2 = time()
    if bp_print_time:
        print(f"Done in {t2-t1:.4f} seconds")

    clear_cupy_mempool=True
    if clear_cupy_mempool:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    import pdb
    #pdb.set_trace()
    print('done')
    bp_astra = bp_astra.astype(np.float32)

    return bp_astra