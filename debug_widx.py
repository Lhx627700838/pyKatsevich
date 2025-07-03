import numpy as np
import pdb
D = 1113
P = 46
R0 = 610
pixel_height = 0.4008525174825175

du = 6.197134545454546e-04
detector_columns_coordinate = du * (np.arange(1376 + 1, dtype=np.float32) 
    + 0.0 # Here "0.0" is a hardcoded value of conf['detector_column_offset']
              - 0.5*(1376 - 1)
              ) 
detector_rows = 144
detector_columns = 1376
detector_row_offset = 0

#print(detector_columns_coordinate)
alpha_m = 0.426235
M = 295
psi_list = np.linspace(-np.pi/2 - alpha_m, np.pi/2 + alpha_m, M)

angles_range = -1
psi_list = -1*psi_list if angles_range<0 else psi_list

detector_rebin_rows= M
wk_index_map = np.zeros((M, detector_columns), dtype=np.float32)
wk_map = np.zeros((M, detector_columns), dtype=np.float32)
# ==== 计算 wk_index_map，只执行一次 ====
tan_psi = np.tan(psi_list)
tan_psi[np.abs(tan_psi) < 1e-6] = 1e-6  # 避免除0
term = psi_list/ tan_psi

for col in range(detector_columns):
    alpha = detector_columns_coordinate[col]
    w_k = (D * P / (2 * np.pi * R0)) * (psi_list * np.cos(alpha) + term * np.sin(alpha))
    w_k_index = -1 * w_k / pixel_height + 0.5 * detector_rows - detector_row_offset
    w_k_index = np.clip(w_k_index, 0.0, detector_rows - 2.001)  # 保证 idx_floor+1 不越界

    wk_map[:, col] = w_k
    wk_index_map[:, col] = w_k_index
print('wk_map')
print(wk_map)
print('wk_index_map')
print(wk_index_map)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 第一个子图: 索引坐标
for k in range(M):
    axes[0].plot(
        np.arange(detector_columns),
        wk_index_map[k, :],
        lw=0.5
    )
axes[0].set_xlabel("Detector column index (alpha_idx)")
axes[0].set_ylabel("Detector row index (w_idx)")
axes[0].set_title("K-curves in detector index coordinates (pixel)")
axes[0].grid(True)
# 颠倒 y 轴，同时保留刻度方向
ylim = axes[0].get_ylim()
axes[0].set_ylim(ylim[::-1])

# 第二个子图: 物理坐标
u_coordinate = du * (
    np.arange(detector_columns, dtype=np.float32)
    + 0.0
    - 0.5 * detector_columns
)

for k in range(M):
    axes[1].plot(
        u_coordinate,
        wk_map[k, :],
        lw=0.5
    )
axes[1].set_xlabel("Detector coordinate u (meters)")
axes[1].set_ylabel("Detector coordinate w (meters)")
axes[1].set_title("K-curves in physical coordinates (mm)")
axes[1].grid(True)


plt.tight_layout()
plt.show()

#pdb.set_trace()