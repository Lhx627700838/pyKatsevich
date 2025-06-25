import numpy as np
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
settings_file = "tests/Naeotom_spine_curve.yaml"
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
'''import matplotlib.pyplot as plt
plt.figure()
plt.imshow(sinogram[0,:,:], cmap='gray')
plt.show()'''

angles_count = 4000
from tests.Naeotom_curve import generate_astra_geom
vol_geom, proj_geom,_ = generate_astra_geom(recon_shape, voxel_size, geom)

conf=create_configuration(
    angles_count,
    geom,
    vol_geom,
    yaml_settings['geometry'].get('options', {})
)

show_td_window = True
w_bottom = np.reshape(conf['proj_row_mins'][:-1], (1, -1))
w_top    = np.reshape(conf['proj_row_maxs'][:-1], (1, -1))
print('w_bottom, w_top', w_bottom, w_top)
dw = conf['pixel_height']
a  = conf['T-D smoothing']
W, U = np.meshgrid(conf['row_coords'][:-1], conf['col_coords'][:-1], indexing='ij')
print(np.shape(W))
# print(f'W = {U}')

def chi(a, U_mesh, W_mesh):
    mask = np.zeros_like(W_mesh)

    # 提前展开上下界区间
    W_top_high     = np.repeat(w_top + a*dw, W_mesh.shape[0], axis=0)
    W_top_low      = np.repeat(w_top - a*dw, W_mesh.shape[0], axis=0)
    W_bottom_high  = np.repeat(w_bottom + a*dw, W_mesh.shape[0], axis=0)
    W_bottom_low   = np.repeat(w_bottom - a*dw, W_mesh.shape[0], axis=0)

    # 中心区域：设为1
    mask[(W_mesh > W_bottom_high) & (W_mesh < W_top_low)] = 1

    # 上过渡区域
    idx_upper = (W_mesh > W_top_low) & (W_mesh < W_top_high)
    mask[idx_upper] = (W_top_high[idx_upper] - W_mesh[idx_upper]) / (2 * a * dw)

    # 下过渡区域
    idx_lower = (W_mesh > W_bottom_low) & (W_mesh < W_bottom_high)
    mask[idx_lower] = (W_mesh[idx_lower] - W_bottom_low[idx_lower]) / (2 * a * dw)

    return mask


TD_mask = chi(a, U, W)
if show_td_window:
    #plt.figure()
    #plt.imshow(TD_mask, cmap='gray')
    #plt.colorbar()
    import tifffile
    tifffile.imwrite('TD_mask.tif',TD_mask)
    # plt.show()