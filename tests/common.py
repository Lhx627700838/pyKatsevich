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

def phantom_objects_3d(
        rows=512,
        columns=512,
        slices=512,
        voxel_size: float=0.04,
        objects_list: list=[]
    ):
    """
    Generate the phantom with +X along columns, +Y along slices, +Z along rows.
    """
    import numpy as np
    def in_sphere(x, y, z, R,
                  x_0=0.0,
                  y_0=0.0,
                  z_0=0.0,
                  in_value=1.0,
                  out_value=0.0
        ):
        sphere_3d = np.ones(shape=(rows, columns, slices))*out_value
        sphere_3d[((x-x_0)**2) + ((y-y_0)**2) + ((z-z_0)**2) < (R**2)] = in_value
        return sphere_3d
    
    def in_cuboid(x, y, z, side,
                  x_0=0.0,
                  y_0=0.0,
                  z_0=0.0,
                  in_value=1.0,
                  out_value=0.0
        ):
        cuboid_3d = np.ones(shape=(rows, columns, slices))*out_value
        cuboid_3d[
            ( (x-x_0) > (-0.5*side) ) & ( (x-x_0) < (0.5*side) ) &
            ( (y-y_0) > (-0.5*side) ) & ( (y-y_0) < (0.5*side) ) &
            ( (z-z_0) > (-0.5*side) ) & ( (z-z_0) < (0.5*side) )
        ] = in_value

        return cuboid_3d

    x_coords = np.arange(
        (1-columns)*0.5*voxel_size,
        columns    *0.5*voxel_size,
        voxel_size,
    )
    y_coords = np.arange(
        (1-rows)*0.5*voxel_size,
        rows    *0.5*voxel_size,
        voxel_size,
    )
    z_coords = np.arange(
        (1-slices)*0.5*voxel_size,
        slices    *0.5*voxel_size,
        voxel_size,
    )

    x_coords = np.arange(
        (1-columns)*0.5*voxel_size,
        columns    *0.5*voxel_size,
        voxel_size,
    )
    y_coords = np.arange(
        (1-rows)*0.5*voxel_size,
        rows    *0.5*voxel_size,
        voxel_size,
    )
    z_coords = np.arange(
        (1-slices)*0.5*voxel_size,
        slices    *0.5*voxel_size,
        voxel_size,
    )

    Y, X, Z = np.meshgrid(y_coords, x_coords, z_coords, indexing="ij")

    phantom_array = np.zeros( (rows, columns, slices) )
    for i, obj in enumerate(objects_list):
        if obj['type'] == 'sphere':
            phantom_array += in_sphere(X, Y, Z, R=obj['R'], x_0=obj['x_0'], y_0=obj['y_0'], z_0=obj['z_0'])
        elif obj['type'] == 'cuboid':
            phantom_array += in_cuboid(X, Y, Z, side=obj['side'], x_0=obj['x_0'], y_0=obj['y_0'], z_0=obj['z_0'])

    return phantom_array

def project(
        volume,
        voxel_size,
        helical_scan_geom
    ):
    """
    Project the volume with the ASTRA toolbox.

    Parameters:
    ===========
    volume : 3D ndarray
        3D image to be projected. Axes order should follow ASTRA's conventions: rows - Y, columns - X, slices - Z.
    voxel_size : float
        Size of voxel side.
    helical_scan_geom : dict
        Geometry settings - dictionary with the following keys:["SOD", "SDD", "detector", "helix"] ( see initialize.create_configuration(...) ).
    """
    import numpy as np
    import astra
    from pykatsevich.geometry import astra_helical_views

    astra_vol_geom = astra.create_vol_geom(
        volume.shape[0],  # rows - Y
        volume.shape[1],  # columns - X
        volume.shape[2],  # slices - Z
        -volume.shape[1]*voxel_size*0.5, volume.shape[1]*voxel_size*0.5,    # along X
        -volume.shape[0]*voxel_size*0.5, volume.shape[0]*voxel_size*0.5,    # along Y
        -volume.shape[2]*voxel_size*0.5, volume.shape[2]*voxel_size*0.5,    # along Z
    )

    if "angles_range" in helical_scan_geom['helix'].keys():
        s_len = helical_scan_geom['helix']['angles_range']
        s_min = -s_len * 0.5
        s_max =  s_len * 0.5
    else:
        s_min = -np.pi + astra_vol_geom['option']['WindowMinZ'] / helical_scan_geom['helix']['pitch_mm_rad']
        s_max =  np.pi + astra_vol_geom['option']['WindowMaxZ'] / helical_scan_geom['helix']['pitch_mm_rad']
        s_len = s_max - s_min

    projs_per_turn = helical_scan_geom['helix']["angles_count"] / s_len * 2*np.pi

    # print(f"projs_per_turn = {projs_per_turn}")

    delta_s = 2*np.pi / projs_per_turn # Turn in radians per projection 

    angles = s_min + delta_s * (np.arange(helical_scan_geom['helix']["angles_count"], dtype=np.float32) + 0.5 )

    # print(f"angles = {angles}")

    ang_step = delta_s

    stride_mm = helical_scan_geom['helix']['pitch_mm_rad'] * ang_step

    views = astra_helical_views(
        helical_scan_geom["SOD"],
        helical_scan_geom["SDD"],
        helical_scan_geom['detector']["detector psize"],
        angles,
        stride_mm
    )

    astra_proj_geom_helical = astra.create_proj_geom(
        "cone_vec",
        helical_scan_geom['detector']["detector rows"],
        helical_scan_geom['detector']["detector cols"],
        views
    )

    # We need to move axes because ASTRA expects the volume axes to be in the following order:
    # Coordinate order: (slice, row, column), or (z, y, x)
    sino_id, sino = astra.create_sino3d_gpu(np.asarray(np.moveaxis(volume, 2, 0), order='C'), astra_proj_geom_helical, astra_vol_geom)
    astra.data3d.clear()

    return sino, astra_vol_geom, astra_proj_geom_helical

def backproject(
        sino_id,
        rec_id
):
    import astra
    algorithm_name = "BP3D_CUDA"
    cfg = astra.astra_dict(algorithm_name)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    backprojection_image = astra.data3d.get(rec_id)

    astra.algorithm.delete([alg_id])

    return backprojection_image

def animate_volume(
        volume,
        axis=0,
        frame_interval=20,
        min_max_per_frame=False,
        colorbar=False,
        title=None,
        **imshow_kwargs
    ):
    import numpy as np

    # make sure it's C-contiguous
    if volume.flags['C_CONTIGUOUS'] != True:
        volume = np.asarray(volume, order='C')

    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation
    from functools import partial

    fig, ax = plt.subplots(1,1)
    if title is not None:
        ax.set_title(title)

    axImage = ax.imshow(
        np.squeeze(np.take(volume, [0], axis=axis)),
        cmap='gray',
        **imshow_kwargs
    )
    if not min_max_per_frame:
        axImage.set(clim=(volume.min(), (volume.max())))

    if colorbar:
        fig.colorbar(axImage, ax=ax)

    def update(imgframe, axImg):
        image = np.squeeze(np.take(volume, [imgframe], axis=axis))
        axImg.set(data=image)
        if min_max_per_frame:
            axImg.set(clim=(image.min(), (image.max())))
        return axImg

    def init_anim():
        return axImage

    anim = FuncAnimation(
        fig,
        partial(update, axImg=axImage),
        frames= range(volume.shape[axis]),
        interval=frame_interval,
        init_func=init_anim,
        blit=False
    )

    plt.show()