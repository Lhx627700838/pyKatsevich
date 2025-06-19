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

from .filter import filter_katsevich, sino_weight_td, backproject_a
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

    if clear_cupy_mempool:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    return bp_astra