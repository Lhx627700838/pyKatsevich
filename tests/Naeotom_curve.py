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
This test runs ASTRA's backprojection on filtered sinogram.
"""
def test_pipeline(settings_file):

    from tests.common import phantom_objects_3d, project, animate_volume, backproject

    import numpy as np
    from matplotlib import pyplot as plt
    from time import time
    import yaml
    import os
    import astra
    import sys
    import tifffile
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from pykatsevich_curve.initialize import create_configuration

    try:
        test_dir = os.path.dirname(os.path.abspath(__file__))
    except:
        print("Failed to get __file__, using current working directory instead")
        test_dir = os.getcwd()

    # 拼接 yaml 文件完整路径
    yaml_path = os.path.join(test_dir, settings_file)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"配置文件未找到: {yaml_path}")

    with open(yaml_path, "r") as file:
        yaml_settings = yaml.safe_load(file)

    recon = yaml_settings['recon']
    recon_shape = (recon['rows'], recon['columns'], recon['slices']) 
    voxel_size = recon['voxel_size']
    geom = yaml_settings['geometry']
    


    # sinogram = tifffile.imread(r"E:\Projects\Liu_proj\pykats\pyKatsevich\scan_001_flat_helix_projections.tif")
    sinogram = tifffile.imread(r"D020_interpolated_Th1_3000_4000.tiff")
    sinogram = sinogram.transpose(0, 2, 1)
    print(sinogram.shape)
    '''import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(sinogram[0,:,:], cmap='gray')
    plt.show()'''
    
    angles_count = sinogram.shape[0]
    vol_geom, proj_geom,_ = generate_astra_geom(recon_shape, voxel_size, geom)
    
    conf=create_configuration(
        angles_count,
        geom,
        vol_geom,
        yaml_settings['geometry'].get('options', {})
    )


    from pykatsevich_curve.reconstruct import reconstruct

    rec_astra = reconstruct(
        sinogram,
        conf,
        vol_geom,
        proj_geom,
        {
            "Diff": {"Progress bar": True, "Print time": True},
            "FwdRebin": {"Progress bar": True, "Print time": True},
            "BackRebin": {"Progress bar": True, "Print time": True},
            "BackProj": {"Progress bar": True, "Print time": True},
        }
    )

    print('done')
    tifffile.imwrite('recon.tif',rec_astra)

def only_reconstruct_pipeline_astra(settings_file):
    import yaml
    import os
    import astra
    import sys
    import tifffile
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from pykatsevich_curve.initialize import create_configuration

    try:
        test_dir = os.path.dirname(os.path.abspath(__file__))
    except:
        print("Failed to get __file__, using current working directory instead")
        test_dir = os.getcwd()

    # 拼接 yaml 文件完整路径
    yaml_path = os.path.join(test_dir, settings_file)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"配置文件未找到: {yaml_path}")

    with open(yaml_path, "r") as file:
        yaml_settings = yaml.safe_load(file)

    recon = yaml_settings['recon']
    recon_shape = (recon['rows'], recon['columns'], recon['slices']) 
    voxel_size = recon['voxel_size']
    geom = yaml_settings['geometry']
    
    filtered_projections = tifffile.imread(r"filtered_proj5.tif")
    vol_geom, proj_geom, lambda_list = generate_astra_geom(recon_shape, voxel_size, geom)

    angles_count = filtered_projections.shape[0]
    conf=create_configuration(
        angles_count,
        geom,
        vol_geom,
        yaml_settings['geometry'].get('options', {})
    )

    from pykatsevich_curve.filter import sino_weight_td, backproject_a
    sino_td = sino_weight_td(filtered_projections, conf, True)

    verbosity_options = {
            "Diff": {"Progress bar": True, "Print time": True},
            "FwdRebin": {"Progress bar": True, "Print time": True},
            "BackRebin": {"Progress bar": True, "Print time": True},
            "BackProj": {"Progress bar": True, "Print time": True},
        }
    
    import tifffile
    tifffile.imwrite('filtered_proj6.tif',sino_td)
    backproject_opts = verbosity_options.get("BackProj", {})
    bp_tqdm_bar = backproject_opts.get("Progress bar", False)
    bp_print_time = backproject_opts.get("Print time", False)

    if bp_print_time and not bp_tqdm_bar:
        print("Backprojection step", end="... ")

    from time import time
    t1 = time()
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

    clear_cupy_mempool=True
    if clear_cupy_mempool:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    print('done')
    tifffile.imwrite('recon.tif',bp_astra)

def only_reconstruct_pipeline(settings_file):
    import yaml
    import os
    import astra
    import sys
    import tifffile
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from pykatsevich_curve.initialize import create_configuration

    try:
        test_dir = os.path.dirname(os.path.abspath(__file__))
    except:
        print("Failed to get __file__, using current working directory instead")
        test_dir = os.getcwd()

    # 拼接 yaml 文件完整路径
    yaml_path = os.path.join(test_dir, settings_file)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"配置文件未找到: {yaml_path}")

    with open(yaml_path, "r") as file:
        yaml_settings = yaml.safe_load(file)

    recon = yaml_settings['recon']
    recon_shape = (recon['rows'], recon['columns'], recon['slices']) 
    voxel_size = recon['voxel_size']
    geom = yaml_settings['geometry']
    
    filtered_sinogram = tifffile.imread(r"filtered_proj6.tif")
    vol_geom, proj_geom, lambda_list = generate_astra_geom(recon_shape, voxel_size, geom)

    angles_count = filtered_sinogram.shape[0]
    conf=create_configuration(
        angles_count,
        geom,
        vol_geom,
        yaml_settings['geometry'].get('options', {})
    )

    from pykatsevich.backproject_gpu import katsevich_backprojection_gpu
    from pykatsevich.backproject import katsevich_backprojection_cpu
    import numpy as np
    # Step 1: 取中间部分投影
    start_idx = 4000
    end_idx = 7000
    filtered_subset = filtered_sinogram[start_idx:end_idx]
    lambda_subset = lambda_list[start_idx:end_idx]

    # Step 2: 计算对应的 z 范围
    P = conf['progress_per_turn']
    z_start = conf['z_min'] + P * (lambda_subset[0] - lambda_list[0]) / (2 * np.pi) #+ P/2
    z_end   = conf['z_min'] + P * (lambda_subset[-1] - lambda_list[0]) / (2 * np.pi) #- P/2
    print('pitch, z_start, z_end', P, z_start, z_end)
    z_min = z_start + P/2
    z_max= z_end - P/2

    conf['z_min'] = z_min
    
    # Step 3: 找出对应体素 z index 范围
    z_vals = conf['z_min'] + np.arange(conf['z_voxels']) * conf['delta_z']
    valid_z = np.where((z_vals >= z_min) & (z_vals <= z_max))[0]
    recon_shape_subset = (conf['x_voxels'], conf['y_voxels'], len(valid_z))

    print("total labmda start:", lambda_list[0])
    print("min max z voxels: ", min(conf['z_voxels'], 1), max(conf['z_voxels'], 1))
    print("min max z vals: ", min(z_vals), max(z_vals))
    print("reconstruction slices number: ", len(valid_z))
    print("reconstruction range z: ", z_min, z_max)
    print("reconstruction range lambda: ", lambda_subset[0], lambda_subset[-1])
    # Step 4: 重建
    '''rec_partial = katsevich_backprojection_gpu(
        filtered_subset,
        conf,
        recon_shape_subset,
        lambda_subset
    )'''

    rec_partial = katsevich_backprojection_cpu(
        filtered_subset,
        conf,
        recon_shape_subset,
        lambda_subset
    )

    print('done')
    tifffile.imwrite('recon.tif', rec_partial)

def generate_astra_geom(volume_shape, voxel_size, helical_scan_geom):
    """
    Generate ASTRA vol_geom and proj_geom from volume shape and helical geometry settings.

    Parameters:
    ===========
    volume_shape : tuple
        Shape of the volume (rows - Y, cols - X, slices - Z)
    voxel_size : float
        Voxel size in mm
    helical_scan_geom : dict
        Dict with keys: ["SOD", "SDD", "detector", "helix"]

    Returns:
    ========
    astra_vol_geom : dict
        ASTRA volume geometry
    astra_proj_geom : dict
        ASTRA projection geometry (cone_vec)
    """
    import numpy as np
    import astra
    from pykatsevich.geometry import astra_helical_views_uv

    rows, cols, slices = volume_shape

    astra_vol_geom = astra.create_vol_geom(
        rows, cols, slices,
        -cols * voxel_size * 0.5, cols * voxel_size * 0.5,
        -rows * voxel_size * 0.5, rows * voxel_size * 0.5,
        -slices * voxel_size * 0.5, slices * voxel_size * 0.5
    )

    if "angles_range" in helical_scan_geom['helix']:
        s_len = helical_scan_geom['helix']['angles_range']
        s_min = -s_len * 0.5
        s_max =  s_len * 0.5
    else:
        s_min = -np.pi + astra_vol_geom['option']['WindowMinZ'] / helical_scan_geom['helix']['pitch_mm_rad']
        s_max =  np.pi + astra_vol_geom['option']['WindowMaxZ'] / helical_scan_geom['helix']['pitch_mm_rad']
        s_len = s_max - s_min

    projs_per_turn = helical_scan_geom['helix']["angles_count"] / (s_len / (2*np.pi))
    delta_s = 2 * np.pi / projs_per_turn
    angles = s_min + delta_s * (np.arange(helical_scan_geom['helix']["angles_count"], dtype=np.float32) + 0.5)
    stride_mm = helical_scan_geom['helix']['pitch_mm_rad'] * delta_s
    
    views = astra_helical_views_uv(
        helical_scan_geom["SOD"],
        helical_scan_geom["SDD"],
        helical_scan_geom['detector']["detector psize u"],
        helical_scan_geom['detector']["detector psize v"],
        angles,
        stride_mm
    )

    astra_proj_geom = astra.create_proj_geom(
        "cone_vec",
        helical_scan_geom['detector']["detector rows"],
        helical_scan_geom['detector']["detector cols"],
        views
    )

    return astra_vol_geom, astra_proj_geom, angles


if __name__=="__main__":
    yaml_settings_file = "Naeotom_spine_10919.yaml"
    test_pipeline(yaml_settings_file)


