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
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize
    import tifffile

    try:
        test_dir = os.path.dirname(os.path.abspath(__file__))
    except:
        print("Failed to get __file__, using current working directory instead")
        test_dir = os.getcwd()

    yaml_path = os.path.join(test_dir, settings_file)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"couldn't find the file: {yaml_path}")

    with open(yaml_path, "r") as file:
        yaml_settings = yaml.safe_load(file)

    phantom_settings = yaml_settings['phantom']
    print(f"Generating the volume with the following objects: {phantom_settings['objects']}")
    voxel_size = phantom_settings['voxel_size']
    # vol_dim = (128, 256, 256)  # (Z, Y, X)

    # # 构建 2D phantom
    # phantom2d = shepp_logan_phantom()
    # phantom2d_resized = resize(phantom2d, (vol_dim[1], vol_dim[2]), mode='reflect', anti_aliasing=True)

    # # 沿 Z 轴堆叠成 3D
    # phantom3d = np.stack([phantom2d_resized] * vol_dim[0], axis=0)  # shape: (128, 128, 128)
    phantom3d = tifffile.imread(r"E:\Projects\Liu_proj\pykats\pyKatsevich\cat_simens.tif")
    print(type(phantom3d))
    print(np.shape(phantom3d))
    
    outputphantom = phantom3d.astype(np.float32)
    tifffile.imwrite('phantom.tif',outputphantom)
    geom = yaml_settings['geometry']
    
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from pykatsevich.initialize import create_configuration

    print("Projecting the phantom", end='...')
    phantom3d = phantom3d.transpose([2,1,0])
    sinogram, vol_geom, proj_geom = project(phantom3d, voxel_size, geom)
    print("Done")

    conf=create_configuration(
        geom,
        vol_geom,
        yaml_settings['geometry'].get('options', {})
    )
    print(np.shape(sinogram))
    sinogram_swapped = sinogram.transpose([1,0,2])

    # plt.figure()
    # plt.imshow(sinogram_swapped[sinogram_swapped.shape[0] // 2], cmap='gray')
    # plt.colorbar()
    # plt.title("Central projection (simulated with ASTRA)")


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

    # im_index = [rec_astra.shape[2] // 4, rec_astra.shape[2] // 2, rec_astra.shape[2] // 4 * 3]
    # n_slices=len(im_index)
    # n_rows = 1
    # fig, ax_array = plt.subplots(n_rows, n_slices)
    # fig.suptitle("Volume slices (ASTRA backprojection)")

    # ax_array = ax_array[np.newaxis, :] if n_rows == 1 else ax_array

    # for i in range(n_slices):
    #     cs=ax_array[0, i].imshow(rec_astra[:, :, im_index[i]], cmap='gray')
    #     ax_array[0, i].set_title(f'Slice {im_index[i]}')
    #     fig.colorbar(cs, ax=ax_array[0, i])

    # plt.show()
    print(np.shape(rec_astra))
    rec_astra = np.transpose(rec_astra,[2,1,0])
    tifffile.imwrite('recon_simcat.tif',rec_astra)

if __name__=="__main__":
    yaml_settings_file = "test04.yaml"
    test_pipeline(yaml_settings_file)