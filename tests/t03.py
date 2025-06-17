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

    phantom_settings = yaml_settings['phantom']
    print(f"Generating the volume with the following objects: {phantom_settings['objects']}")
    voxel_size = phantom_settings['voxel_size']
    phantom  = phantom_objects_3d(
        phantom_settings['rows'], phantom_settings['columns'], phantom_settings['slices'],
        voxel_size=voxel_size, objects_list=phantom_settings['objects'])

    geom = yaml_settings['geometry']
    
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from pykatsevich.initialize_orig import create_configuration

    print("Projecting the phantom", end='...')
    sinogram, vol_geom, proj_geom = project(phantom, voxel_size, geom)
    print("Done")

    conf=create_configuration(
        geom,
        vol_geom,
        yaml_settings['geometry'].get('options', {})
    )

    sinogram_swapped = np.asarray(np.swapaxes(sinogram, 0, 1), order='C')

    plt.figure()
    plt.imshow(sinogram_swapped[sinogram_swapped.shape[0] // 2], cmap='gray')
    plt.colorbar()
    plt.title("Central projection (simulated with ASTRA)")


    from pykatsevich.reconstruct import reconstruct

    rec_astra = reconstruct(
        sinogram_swapped,
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

    im_index = [rec_astra.shape[2] // 4, rec_astra.shape[2] // 2, rec_astra.shape[2] // 4 * 3]
    n_slices=len(im_index)
    n_rows = 1
    fig, ax_array = plt.subplots(n_rows, n_slices)
    fig.suptitle("Volume slices (ASTRA backprojection)")

    ax_array = ax_array[np.newaxis, :] if n_rows == 1 else ax_array

    for i in range(n_slices):
        cs=ax_array[0, i].imshow(rec_astra[:, :, im_index[i]], cmap='gray')
        ax_array[0, i].set_title(f'Slice {im_index[i]}')
        fig.colorbar(cs, ax=ax_array[0, i])

    plt.show()

if __name__=="__main__":
    yaml_settings_file = "test03.yaml"
    test_pipeline(yaml_settings_file)