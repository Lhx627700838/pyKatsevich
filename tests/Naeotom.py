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

    from common import phantom_objects_3d, project, animate_volume, backproject

    import numpy as np
    from matplotlib import pyplot as plt
    from time import time
    import yaml
    import os
    import astra
    import sys
    import tifffile
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from pykatsevich.initialize import create_configuration

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
    sinogram = tifffile.imread(r"E:\Projects\Liu_proj\pykats\pyKatsevich\naeotom_cat_10919.tif")
    vol_geom, proj_geom = generate_astra_geom(recon_shape, voxel_size, geom)

    conf=create_configuration(
        geom,
        vol_geom,
        yaml_settings['geometry'].get('options', {})
    )


    from pykatsevich.reconstruct import reconstruct
    sinogram = sinogram[:, ::-1, :]
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
    rec_astra = np.transpose(rec_astra,[2,1,0])
    tifffile.imwrite('recon_Naeotom.tif',rec_astra)

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
    
    return astra_vol_geom, astra_proj_geom


if __name__=="__main__":
    yaml_settings_file = "Naeotom_spine_10919.yaml"
    test_pipeline(yaml_settings_file)


