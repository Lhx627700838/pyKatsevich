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

    col_coords = conf['col_coords'][:-2]
    row_coords = conf['row_coords'][:-2]

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

    range_object = tqdm(range(0, sinogram.shape[0]-1), "Derivatives   ") if tqdm_imported and tqdm_bar else range(0, sinogram.shape[0]-1)

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

def fw_height_rebinning(
        input_array,
        conf, # Helical scan configuration helpful for the katsevich formula
        tqdm_bar=False
    ):
    """
    Forward height rebinning of the derivatives array.
    
    Parameters:
    ===========
    input_array : ndarray of float
        The 3D array with projection data derivatives. The expected shape is (views, det_rows, det_cols).
    conf : dict
        Dictionary with the helical configuration used in the katsevich formula. See initialize.create_configuration() for details.
    tqdm_bar : bool
        Show tqdm progress bar (ignored if tqdm is not installed). The defauls it False.

    Returns:
    ========
    output_array : ndarray
        Rebinned array of derivatives.
    """

    try:
        from tqdm import tqdm
        tqdm_imported = True
    except:
        tqdm_imported = False

    pixel_height = conf['pixel_height']
    detector_rows = conf['detector rows']
    detector_columns = conf['detector cols']
    detector_row_offset = 0 # Hardcoded value for conf['detector_row_offset']
    
    fwd_rebin_row = conf['fwd_rebin_row']
    
    output_array = np.zeros(shape=((input_array.shape[0],conf['detector_rebin_rows'], detector_columns)))

    range_object = tqdm(range(0, input_array.shape[0]), "Forward rebin.") if tqdm_imported and tqdm_bar else range(0, input_array.shape[0])

    for proj_index in range_object:
        proj = proj_index - 0
        for col in range(detector_columns):

            # Map scaled coordinates into original row index range of integers
            # for rebinning
            # Sign is inverted for shift and thus also for offset.

            # TODO: rows-1 here too?

            row_scaled = fwd_rebin_row[:, col] / pixel_height + 0.5 \
                * detector_rows - detector_row_offset

            # make sure row and row+1 are in valid row range

            np.clip(row_scaled, 0, detector_rows - 2, row_scaled)

            # we need integer indexes (this repeated creation may be slow)

            row = np.floor(row_scaled).astype(np.int32)

            # linear interpolation of row neighbors

            row_frac = row_scaled - row
            output_array[proj, :, col] = (1 - row_frac) \
                * input_array[proj, row, col] + row_frac \
                * input_array[proj, row + 1, col]

    return output_array

def compute_hilbert_kernel(
        conf
    ):
    proj_filter_array = np.zeros(conf['kernel_width'], dtype=np.float32)

    # We use a simplified hilbert kernel for now

    kernel_radius = conf['kernel_radius']
    for i in range(conf['kernel_width']):
        proj_filter_array[i] = (1.0 - np.cos(np.pi * (i - kernel_radius - 0.5))) \
                            / (np.pi * (i - kernel_radius - 0.5))
    
    return proj_filter_array


def hilbert_conv(
        input_array,
        hilbert_array,
        conf
    ):

    from numpy import convolve

    detector_columns = conf['detector cols']
    detector_rebin_rows = conf['detector_rebin_rows']

    output_array = np.zeros_like(input_array)
    # TODO: use rectangular hilbert window as suggested in Noo paper?

    for proj_index in range(0, input_array.shape[0]):
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

def hilbert_trans_scipy(input_array):
    from scipy.signal import hilbert
    output_array = np.imag(hilbert(input_array, axis=2))
    return output_array

def rev_rebin_vec(
        input_array,
        conf,
        tqdm_bar=False
    ):
    """
    Run vectorized version of reverse height rebinning step on projections.
    Expects input_array to be on (projections, rebin, columns) form.
    Parameters
    ----------
    input_array : ndarray
        Input array.
    conf : dict
        Configuration dictionary.
    tqdm_bar : bool
        Show tqdm progress bar (ignored if tqdm is not installed). The defauls it False.

    Returns
    -------
    output : float
        Reverse rebinning time.
    """
    try:
        from tqdm import tqdm
        tqdm_imported = True
    except:
        tqdm_imported = False

    detector_rows = conf['detector rows']
    detector_columns = conf['detector cols']
    detector_column_offset = 0 # conf['detector_column_offset']
    detector_rebin_rows = conf['detector_rebin_rows']
    detector_column_offset = 0 # conf['detector_column_offset']
    row_coords = conf['row_coords'] [:-1]
    fwd_rebin_row = conf['fwd_rebin_row']
    rebin_row = conf['rebin_row']
    fracs_0, fracs_1 = conf['rebin_fracs_0'], conf['rebin_fracs_1']

    output_array = np.zeros(
        shape=(
            input_array.shape[0],
            detector_rows,
            detector_columns
        )
    )
    (src, dst) = (input_array, output_array)

    pos_start = int(0.5 * detector_columns - detector_column_offset)

    range_object = tqdm(range(0, input_array.shape[0]), "Reverse rebin.") if tqdm_imported and tqdm_bar else range(0, input_array.shape[0])

    for proj_index in range_object:
        proj = proj_index - 0

        for col in range(pos_start, detector_columns):
            dst[proj, :, col] = fracs_1[:, col] * src[proj, rebin_row[:, col], col] + fracs_0[:, col] * src[proj, rebin_row[:, col] + 1, col]

        for col in range(pos_start):
            dst[proj, :, col] = fracs_1[:, col] * src[proj, rebin_row[:, col] - 1, col] + fracs_0[:, col] * src[proj, rebin_row[:, col], col]

    return dst

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
        print(f"Done in {t2-t1:.4f} seconds")

    fwd_rebin_opts = verbosity_options.get("FwdRebin", {})
    fwd_rebin_tqdm_bar = fwd_rebin_opts.get("Progress bar", False)
    fwd_rebin_time = fwd_rebin_opts.get("Print time", False)

    if fwd_rebin_time and not fwd_rebin_tqdm_bar:
        print("Forward height rebinning step", end="... ")
    t1 = time()
    fwd_rebin_array = fw_height_rebinning(diff_proj, conf, fwd_rebin_tqdm_bar)
    t2 = time()
    if fwd_rebin_time:
        print(f"Done in {t2-t1:.4f} seconds")

    hilbert_array = compute_hilbert_kernel(conf)
    sino_heilbert_trans = hilbert_conv(fwd_rebin_array, hilbert_array, conf)

    back_rebin_opts = verbosity_options.get("BackRebin", {})
    back_rebin_tqdm_bar = back_rebin_opts.get("Progress bar", False)
    back_rebin_time = back_rebin_opts.get("Print time", False)

    if back_rebin_time and not back_rebin_tqdm_bar:
        print("Backward height rebinning step", end="... ")
    t1 = time()
    filtered_projections = rev_rebin_vec(sino_heilbert_trans, conf, back_rebin_tqdm_bar)
    t2 = time()
    if back_rebin_time:
        print(f"Done in {t2-t1:.4f} seconds")

    return filtered_projections

def flat_backproject_chunk(
        input_array,
        conf,
        tqdm_bar=False
    ):
    """
    Run backprojection on chunk of projections keeping the results in
    output_array.

    Parameters
    ----------
    input_array : ndarray
        Input array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Backprojection time.
    """

    import time

    try:
        from tqdm import tqdm
        tqdm_imported = True
    except:
        tqdm_imported = False

    time_before = time.time()

    # We're taking all projections:
    first_proj = 0
    last_proj = input_array.shape[0] - 1

    # reconstruct for all z-slices:
    first_z = 0
    last_z = conf['z_voxels'] - 1

    # Limit to actual projection sources in chunk

    source_pos = conf['source_pos'][first_proj:last_proj + 1]

    scan_radius = conf['scan_radius']
    scan_diameter = conf['scan_diameter']
    x_min = conf['x_min']
    x_max = conf['x_max']
    y_min = conf['y_min']
    y_max = conf['y_max']
    z_min = conf['z_min']
    delta_x = conf['delta_x']
    delta_y = conf['delta_y']
    delta_z = conf['delta_z']
    x_voxels = conf['x_voxels']
    y_voxels = conf['y_voxels']
    z_voxels = conf['z_voxels']
    fov_radius = conf['fov_radius']
    pixel_span = conf['detector_pixel_span']
    pixel_height = conf['detector_pixel_height']
    detector_rows = conf['detector rows']
    detector_columns = conf['detector cols']
    detector_row_offset = 0
    detector_column_offset = 0 # conf['detector_column_offset']
    detector_row_shift = -detector_row_offset + 0.5 * (conf['detector rows'] - 1)
    progress_per_radian = conf['progress_per_radian']

    row_mins_array = conf['proj_row_mins']
    row_maxs_array = conf['proj_row_maxs']

    output_array = np.zeros( shape = (y_voxels, x_voxels, z_voxels) )

    (prev_proj, cur_proj, next_proj) = range(3)

    # Calculate x, y and squared coordinates once and for all

    x_coords = np.arange(x_voxels, dtype=np.float32) * delta_x + x_min + 0.5*delta_x
    # x_coords = x_max - 0.5*delta_x - np.arange(x_voxels, dtype=np.float32) * delta_x
    sqr_x_coords = x_coords ** 2
    # y_coords = y_max - 0.5*delta_y - np.arange(y_voxels, dtype=np.float32) * delta_y
    y_coords = np.arange(y_voxels, dtype=np.float32) * delta_y + y_min +0.5*delta_y
    # y_coords = np.arange(y_voxels, dtype=np.float32) * delta_y + y_min
    
    sqr_y_coords = y_coords ** 2
    rad_sqr = fov_radius ** 2
    proj_row_coords = np.zeros(3, dtype=np.float32)
    proj_row_steps = np.zeros(3, dtype=np.float32)
    (debug_x, debug_y) = (x_voxels // 3, y_voxels // 3)
    
    range_object = tqdm(range(y_voxels), "Backprojection") if tqdm_imported and tqdm_bar else range(y_voxels)
    
    for y in range_object:
        y_coord = y_coords[y]
        y_coord_sqr = sqr_y_coords[y]
        for x in range(x_voxels):
            x_coord = x_coords[x]
            x_coord_sqr = sqr_x_coords[x]

            # Ignore voxels with center outside the cylinder with fov_radius

            if x_coord_sqr + y_coord_sqr > rad_sqr:
                continue

            # Constant helper arrays for this particular (x,y) and all angles.
            # Column and scale helpers remain constant for all z-values but
            # row helpers must be calculated for each z.

            # The 'denominator' or scaling function

            scale_helpers = scan_radius - x_coord * np.cos(source_pos) \
                - y_coord * np.sin(source_pos)

            # Projected (float) coordinates from column projection formula

            proj_col_coords = scan_diameter * (-x_coord*np.sin(source_pos) + y_coord * np.cos(source_pos)) \
                / scale_helpers

            # Matching column indices in exact (float) and whole (integer) form
            # We divide signed coordinate by size and shift by half the number
            # of pixels to get unsigned pixel index when rounding to integer.
            # We need to either round towards zero or limit range to actual
            # index range to avoid hits exactly on the borders to result in out
            # of bounds index.
            # Sign is inverted for shift and thus also for offset.

            # TODO: columns-1 here too?

            proj_col_reals = proj_col_coords / pixel_span + 0.5 \
                * detector_columns - detector_column_offset
            proj_col_ints = proj_col_reals.astype(np.int32)

            # TODO: clip to det-2 here? (we do that later anyway)

            np.clip(proj_col_ints, 0, detector_columns-1, proj_col_ints)
            proj_col_fracs = proj_col_reals - proj_col_ints

            # Row coordinate step for each z increment: this equals the
            # derivative with respect to z of the row projection formula

            proj_row_coord_diffs = scan_diameter / scale_helpers

            # Row index step for each z index increment: same as above but
            # scaled to be in z index instead of coordinate

            proj_row_ind_diffs = proj_row_coord_diffs * delta_z \
                / pixel_height

            # Row coordinates for z_min using row coordinate formula.
            # Used to calculate the row index for any z index in the z loop

            proj_row_coord_z_min = scan_diameter * (z_min
                    - progress_per_radian * source_pos) / scale_helpers
            proj_row_ind_z_min = proj_row_coord_z_min / pixel_height \
                + detector_row_shift

            # Interpolate nearest precalculated neighbors in limit row coords.
            # They are used as row coordinate boundaries for z loop and in
            # boundary weigths.
            # Please note that row_mins/maxs are built from the extended
            # col coords so that they include one extra element to allow this
            # interpolation even for the last valid column index,
            # (detector_columns-1)

            proj_row_coord_mins = (1 - proj_col_fracs) \
                * row_mins_array[proj_col_ints] + proj_col_fracs \
                * row_mins_array[proj_col_ints + 1]
            proj_row_coord_maxs = (1 - proj_col_fracs) \
                * row_maxs_array[proj_col_ints] + proj_col_fracs \
                * row_maxs_array[proj_col_ints + 1]

            # Use row projection formula to calculate z limits from row limits

            z_coord_mins = source_pos * progress_per_radian \
                + proj_row_coord_mins * scale_helpers / scan_diameter
            z_coord_maxs = source_pos * progress_per_radian \
                + proj_row_coord_maxs * scale_helpers / scan_diameter

            # Extract naive integer indices - handle out of bounds later
            # We round inwards using ceil and floor respectively to avoid
            # excess contribution from border pixels

            z_firsts = np.ceil( (z_coord_mins - z_min) / delta_z ).astype(np.int32)
            z_lasts = np.floor( (z_coord_maxs - z_min) / delta_z ).astype(np.int32)
            
            for proj_index in range(first_proj, last_proj):
                proj = proj_index - first_proj

                # Reset proj_row_coords triple to first row coordinates before
                # each z loop.
                # Please note that proj_row_coords prev and next values are
                # *only* used for projs where prev and next makes sense below.
                # So we just ignore the values for out of bounds border cases.

                proj_row_coords[:] = 0
                proj_row_steps[:] = 0
                if proj > 0:
                    proj_row_steps [prev_proj] = proj_row_coord_diffs[proj - 1] * delta_z
                    proj_row_coords[prev_proj] = proj_row_coord_z_min[proj - 1] + z_firsts[proj] * proj_row_steps[prev_proj]
                proj_row_steps[cur_proj] = proj_row_coord_diffs[proj] * delta_z
                proj_row_coords[cur_proj] = proj_row_coord_z_min[proj] \
                    + z_firsts[proj] * proj_row_steps[cur_proj]
                if proj < last_proj - 1:
                    proj_row_steps[next_proj] = \
                        proj_row_coord_diffs[proj + 1] * delta_z
                    proj_row_coords[next_proj] = \
                        proj_row_coord_z_min[proj + 1] + z_firsts[proj] \
                        * proj_row_steps[next_proj]
                if (z_coord_maxs[proj] < (z_min + first_z * delta_z)) or (z_coord_mins[proj] > (z_min + last_z * delta_z)):
                    continue
                # if x == debug_x and y == debug_y:
                #     print('loop (%d, %d, %d:%d) proj %d %f:%f'
                #                   % (
                #         x,
                #         y,
                #         z_firsts[proj],
                #         z_lasts[proj],
                #         proj_index,
                #         proj_row_coord_mins[proj] / pixel_height + 
                #                       detector_row_shift,
                #         proj_row_coord_maxs[proj] / pixel_height +
                #                       detector_row_shift,
                #         ))

                # Include last z index

                for z in range(z_firsts[proj], z_lasts[proj] + 1):

                    # Always update projected row coordinates

                    (prev_row_coord, cur_row_coord, next_row_coord) = \
                        proj_row_coords[:]
                    proj_row_coords += proj_row_steps

                    # Skip out of bounds indices here to avoid border weighting
                    # on boundary clipped index values

                    if (z < first_z) or (z > last_z):
                        continue
                    z_coord = z_min + z * delta_z
                    z_local = z - first_z

                    # Border weight only applies for first and last z

                    if (z == z_firsts[proj]) and next_row_coord \
                        < proj_row_coord_mins[proj + 1]:
                        weight = 0.5 + (z_coord - z_coord_mins[proj]) \
                            / (z_coord_mins[proj + 1]
                               - z_coord_mins[proj])
                        # if x == debug_x and y == debug_y:
                        #     print('first weight: %f %f %f %f %f'
                        #              % (next_row_coord,
                        #             proj_row_coord_mins[proj + 1],
                        #             z_coord_mins[proj + 1], z_coord,
                        #             z_coord_mins[proj]))
                    elif (z == z_lasts[proj]) and prev_row_coord \
                        > proj_row_coord_maxs[proj - 1]:
                        weight = 0.5 + (z_coord_maxs[proj] - z_coord) \
                            / (z_coord_maxs[proj] - z_coord_maxs[proj
                               - 1])
                        # if x == debug_x and y == debug_y:
                        #     print('last weight: %f %f %f %f %f'
                        #             % (prev_row_coord,
                        #             proj_row_coord_mins[proj - 1],
                        #             z_coord_maxs[proj], z_coord,
                        #             z_coord_maxs[proj - 1]))
                    else:
                        weight = 1.0

                    # TODO: is this correct? (0.5 less than direct proj coord)
                    # ... obviously from the (detector_rows-1) in
                    # proj_row_ind_z_min
                    # Removing -1 breaks result, inspect same -1 for col?
                    # Row indices in exact (real) and whole (integer) form.
                    # Offset is already included in proj_row_ind_z_min

                    proj_row_real = (proj_row_ind_z_min[proj] + z \
                        * proj_row_ind_diffs[proj])
                    proj_row_int = proj_row_real.astype(np.int32)

                    # make sure row and row+1 are in valid row range

                    proj_row_int = min(max(proj_row_int, 0), detector_rows - 2)
                    proj_row_frac = proj_row_real - proj_row_int
                    proj_row_int_next = proj_row_int + 1
                    proj_col_int = proj_col_ints[proj]

                    # make sure col and col+1 are in valid col range

                    proj_col_int = min(max(proj_col_int, 0), detector_columns - 2)
                    proj_col_frac = proj_col_fracs[proj]
                    proj_col_int_next = proj_col_int + 1
                    proj_mean = input_array[proj, proj_row_int, proj_col_int] * (1 - proj_row_frac) * (1 - proj_col_frac) \
                              + input_array[proj, proj_row_int_next, proj_col_int] * proj_row_frac  * (1 - proj_col_frac) \
                              + input_array[proj, proj_row_int, proj_col_int_next] * (1 - proj_row_frac) * proj_col_frac  \
                              + input_array[proj, proj_row_int_next, proj_col_int_next] * proj_row_frac * proj_col_frac
                    
                    contrib = proj_mean * weight  / scale_helpers[proj]
                    output_array[ y, x, z_local] += contrib
                    # if x == debug_x and y == debug_y: # and z_local==debug_z:
                    #     print('update (%d, %d, %d): %f (%f) from %d'
                    #              % (
                    #         x,
                    #         y,
                    #         z_local, #z,
                    #         contrib,
                    #         output_array[x, y, z_local],
                    #         proj_index,
                    #         ))
                        # print('w %f r %d %f (%f) c %d %f (%f) m %f'
                        #          % (
                        #     weight,
                        #     proj_row_int,
                        #     proj_row_frac,
                        #     proj_row_real,
                        #     proj_col_int,
                        #     proj_col_frac,
                        #     proj_col_reals[proj],
                        #     proj_mean,
                        #     ))
    time_after = time.time()
    time_backproj = time_after - time_before
    print(f'Finished backproject kernel in {time_backproj:.2f} sec.')

    # The result is scaled with delta_s/(2*pi) = projs_per_turn:
    return output_array / conf['projs_per_turn']

def sino_weight_td(input_array, conf, show_td_window=False):
    w_bottom = np.reshape(conf['proj_row_mins'][:-1], (1, -1))
    w_top    = np.reshape(conf['proj_row_maxs'][:-1], (1, -1))

    dw = conf['detector rows'] * conf['pixel_height']
    a  = conf['T-D smoothing']
    W, U = np.meshgrid(conf['row_coords'][:-1], conf['col_coords'][:-1], indexing='ij')

    # print(f'W = {U}')

    def chi(a, U_mesh, W_mesh):
        # TODO: move this function to the module scope
        mask = np.zeros(shape=(conf['detector rows'], conf['detector cols']))

        W_top_high= np.repeat(w_top+a*dw, W_mesh.shape[0], axis=0)
        W_top_low = np.repeat(w_top-a*dw, W_mesh.shape[0], axis=0)
        W_bottom_high = np.repeat(w_bottom+a*dw, W_mesh.shape[0], axis=0)
        W_bottom_low  = np.repeat(w_bottom-a*dw, W_mesh.shape[0], axis=0)

        # plt.figure()
        # plt.imshow((W_top_low < W_mesh) & (W_mesh < W_top_high))
        # plt.title("Top of the mask")

        mask[(W_top_low < W_mesh) & (W_mesh < W_top_high)] = ((W_top_high - W_mesh) / (2*a*dw))[(W_top_low < W_mesh) & (W_mesh < W_top_high)]

        mask[(W_bottom_high < W_mesh) & (W_mesh < W_top_low)] = 1

        mask[(W_bottom_low < W_mesh) & (W_mesh < W_bottom_high)] = ((W_mesh - W_bottom_low) / (2*a*dw))[(W_bottom_low < W_mesh) & (W_mesh < W_bottom_high)]

        # for j in range(W_mesh.shape[0]):
        #     w_bottom+a*dw = 1 * (((w_top-a*dw) > W_mesh[j]) & ((w_bottom+a*dw) < W_mesh[j]))
            
            # mask[j] = 1 * ((w_bottom-a*dw) > W_mesh[j])

        return mask
    
    TD_mask = chi(a, U, W)
    if show_td_window:
        plt.figure()
        plt.imshow(TD_mask, cmap='gray')
        plt.colorbar()
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

                float scale_coeff = scale_const*(scan_radius - X*ang_cos - Y*ang_sin);
                    
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