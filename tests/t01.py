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
A set qualitative tests that output images after each step of Katsevich filtering.
In the end, both Numpy and ASTRA backprojections are run on the filtered data.
"""
def test_pipeline(settings_file):

    from common import phantom_objects_3d, project, animate_volume, backproject

    import numpy as np
    from matplotlib import pyplot as plt
    from time import time
    import yaml
    import os
    import astra

    test_dir = os.getcwd()
    try:
        test_dir = os.sep.join(__file__.split("/")[:-1])
    except:
        print("Failed to pick the path to the Python file, picking current work directory instead")

    yaml_settings = {}
    with open(os.sep.join([test_dir, settings_file]), "r") as file:
        yaml_settings = yaml.safe_load(file)

    phantom_settings = yaml_settings['phantom']
    print(f"Generating the volume with the following objects: {phantom_settings['objects']}")
    voxel_size = phantom_settings['voxel_size']
    phantom  = phantom_objects_3d(
        phantom_settings['rows'], phantom_settings['columns'], phantom_settings['slices'],
        voxel_size=voxel_size, objects_list=phantom_settings['objects'])

    animate_volume(
        phantom,
        axis=2,
        colorbar=True,
        min_max_per_frame=False,
        title="Volume to be projected"
    )

    geom = yaml_settings['geometry']
    
    from pykatsevich.initialize import create_configuration

    print("Projecting the phantom", end='...')
    sinogram, vol_geom, proj_geom = project(phantom, voxel_size, geom)
    print("Done")

    # animate_volume(sinogram, axis=1, colorbar=True,title='ASTRA projections')

    conf=create_configuration(
        geom,
        vol_geom,
        yaml_settings['geometry'].get('options', {})
    )

    # sinogram_swapped = np.asarray(np.flip(np.swapaxes(sinogram, 0, 1), axis=2), order='C')
    sinogram_swapped = np.asarray(np.swapaxes(sinogram, 0, 1), order='C')

    plt.figure()
    plt.imshow(sinogram_swapped[sinogram_swapped.shape[0] // 2], cmap='gray')
    plt.colorbar()
    plt.title("Central projection (simulated with ASTRA)")

    animate_volume(sinogram_swapped, axis=0, colorbar=True)

    from pykatsevich.filter import differentiate, fw_height_rebinning, rev_rebin_vec

    print("Computing derivatives", end="...", flush=True)
    t1 = time()
    sino_diff = differentiate(sinogram_swapped, conf)
    t2 = time()
    print(f"Done in {t2-t1:.2f} sec.")

    plt.figure()
    plt.imshow(sino_diff[sino_diff.shape[0] // 2], cmap='gray')
    plt.colorbar()
    plt.title("Central projection after computing the Noo's derivatives")

    animate_volume(sino_diff, axis=0, colorbar=True, min_max_per_frame=False)

    print("Forward rebinning", end="...", flush=True)
    t1=time()
    sino_rebin = fw_height_rebinning(sino_diff, conf)
    t2=time()
    print(f"Done in {t2-t1:.2f} sec.")

    fig, ax = plt.subplots(figsize=(6.2, 6))
    cs=ax.imshow(sino_rebin[sino_rebin.shape[0] // 2], cmap='gray', aspect='equal')
    fig.colorbar(cs, ax=ax)
    ax.set_title("Central projection after rebinning")

    # # Plot K-lines
    plt.figure()
    for i in range(conf['detector_rebin_rows']):
        plt.plot(conf['col_coords'][:-1], conf['fwd_rebin_row'][i, :], color='gray' )
    plt.plot(conf['col_coords'], conf['proj_row_mins'], color='red' )
    plt.plot(conf['col_coords'], conf['proj_row_maxs'], color='green' )
    plt.xlabel('u, mm')
    plt.ylabel(r'$w_{\kappa}$, mm')
    # plt.ylim([ conf['row_coords'][0], conf['row_coords'][-1] ])
    plt.plot([conf['col_coords'][0], conf['col_coords'][-1]], [conf['row_coords'][0], conf['row_coords'][0]] , color='blue')
    plt.plot([conf['col_coords'][0], conf['col_coords'][-1]], [conf['row_coords'][-1], conf['row_coords'][-1]], color='blue')
    print(f"half FOV fan angle = {conf['half_fan_angle']:.2f} rad.")
    print(f"FOV radius = {conf['fov_radius']}")

    from pykatsevich.filter import compute_hilbert_kernel, hilbert_conv, hilbert_trans_scipy

    print("1D Hilbert transform", end="...", flush=True)
    t1=time()
    hilbert_array = compute_hilbert_kernel(conf)
    # sino_heilbert_trans = hilbert_trans_scipy(sino_rebin)
    sino_heilbert_trans = hilbert_conv(sino_rebin, hilbert_array, conf)
    t2=time()
    print(f"Done in {t2-t1:.2f} sec.")

    # plt.figure()
    # plt.plot(hilbert_array)
    # plt.title("Helbert kernel")

    plt.figure()
    plt.imshow(sino_heilbert_trans[sino_heilbert_trans.shape[0] // 2], cmap='gray')
    plt.colorbar()
    plt.title("Central projection after 1D Hilbert transorm")

    # animate_volume(sino_heilbert_trans, axis=0, colorbar=True)

    print("Reverse rebinning", end="...\n", flush=True)
    t1=time()
    sino_reverse_rebin = rev_rebin_vec(sino_heilbert_trans, conf)
    t2=time()
    print(f"Done in {t2-t1:.2f} sec.")

    plt.figure()
    plt.imshow(sino_reverse_rebin[sino_reverse_rebin.shape[0] // 2], cmap='gray')
    plt.colorbar()
    plt.title("Central projection after reverse rebinning")

    # animate_volume(sino_reverse_rebin, axis=0, colorbar=True)

    # Backprojeciton with astra
    from pykatsevich.filter import sino_weight_td
    sino_td = sino_weight_td(sino_reverse_rebin, conf, False)

    animate_volume(sino_td, axis=0, colorbar=True, aspect='equal')

    from pykatsevich.filter import backproject_a

    bp_astra = backproject_a(sino_td, conf, vol_geom, proj_geom)

    rec_astra = bp_astra

    # plt.figure()
    # plt.imshow(rec_astra[:, :, rec_astra.shape[0] // 4], cmap='gray')
    # plt.colorbar()
    # plt.title("Volume slice (ASTRA backprojection)")
    # animate_volume(rec_astra, axis=2, colorbar=True)

    print(f"Running CPU backprojection prototype")
    from pykatsevich.filter import flat_backproject_chunk
    rec_proto = flat_backproject_chunk(sino_reverse_rebin, conf)
    print("Done")

    n_subplots=3
    im_index = [rec_proto.shape[2] // 4, rec_proto.shape[2] // 2, rec_proto.shape[2] // 4 * 3]
    fig, ax_array = plt.subplots(3, n_subplots, sharex=True, sharey=True)
    for i in range(n_subplots):
        cs=ax_array[0, i].imshow(phantom[:, :, im_index[i]], cmap='gray')
        ax_array[0, i].set_title(f'Phantom')
        fig.colorbar(cs, ax=ax_array[0, i])
        cs=ax_array[1, i].imshow(rec_proto[:, :, im_index[i]], cmap='gray')
        ax_array[1, i].set_title(f'backprojection CPU')
        fig.colorbar(cs, ax=ax_array[1, i])
        cs=ax_array[2, i].imshow(rec_astra[:, :, im_index[i]], cmap='gray')
        ax_array[2, i].set_title(f'backprojection ASTRA (GPU)')
        fig.colorbar(cs, ax=ax_array[2, i])

    fig.suptitle("Volume slices")

    # animate_volume(rec_proto, axis=2, colorbar=True)

    plt.show()

if __name__=="__main__":
    yaml_settings_file = "test03.yaml"
    test_pipeline(yaml_settings_file)