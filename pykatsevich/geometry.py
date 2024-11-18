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

def astra_helical_views(
        SOD: float,
        SDD: float,
        pixel_size: float,
        angles: np.ndarray,
        pitch_per_angle:float
    ):
    """
    Generate ASTRA views from the helix description.

    Parameters:
    ===========
    SOD : float
        Source-objet distance.
    SDD : float
        Source-detector distance.
    pixel_size : float
        Size of detetor pixels.
    angles : np.ndarray
        Array of projection angles, counter-clockwise rotation around Z.
        Each value is the angle between the X-axis and projection direction.
    pitch_per_angle : float
        Vertical pitch size per every projection angle in chosen units, e.g., mm.
    
    Return:
    =======
        Array of 12-element vectors, each vector describing a single projection view.
    """
    rot = lambda x, theta: [x[0]*np.cos(theta)-x[1]*np.sin(theta),x[0]*np.sin(theta)+x[1]*np.cos(theta),x[2]]

    vertical_shift = np.linspace(
        -pitch_per_angle*angles.shape[0]*0.5,
        pitch_per_angle*angles.shape[0]*0.5,
        angles.shape[0]
    )

    views_list = []

    start_view = [SOD, 0, 0, -(SDD - SOD), 0, 0, 0, pixel_size, 0, 0, 0, pixel_size]

    for i, aa in enumerate(angles):
        views_list.append(np.concatenate((
            rot(start_view[0:3], aa) + np.array([0, 0, vertical_shift[i]]),
            rot(start_view[3:6], aa) + np.array([0, 0, vertical_shift[i]]),
            rot(start_view[6:9], aa),
            rot(start_view[9:12],aa)
        )))

    views_array = np.asarray(views_list)
    return views_array