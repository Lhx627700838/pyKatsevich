import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def draw_box(ax, corners, color='green', alpha=0.2):
    verts = [
        [corners[0], corners[1], corners[2], corners[3]],
        [corners[4], corners[5], corners[6], corners[7]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[2], corners[3], corners[7], corners[6]],
        [corners[1], corners[2], corners[6], corners[5]],
        [corners[4], corners[7], corners[3], corners[0]]
    ]
    ax.add_collection3d(Poly3DCollection(verts, facecolors=color, linewidths=0.5, edgecolors='k', alpha=alpha))

def set_axes_equal(ax):
    """统一x/y/z轴比例"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range]) / 2.0
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def visualize_system_dynamic(fig, ax, image, source, detector, source_traj, detector_center_traj):
    ax.clear()
    ax.set_title("CT Geometry Visualization")

    # --- 体素区域（绿色盒子） ---
    voxel_coords = image.voxelCoordinate  # (3, x, y, z)
    x_vals = voxel_coords[0, :, :, :].ravel()
    y_vals = voxel_coords[1, :, :, :].ravel()
    z_vals = voxel_coords[2, :, :, :].ravel()
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    z_min, z_max = np.min(z_vals), np.max(z_vals)

    voxel_corners = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ])
    draw_box(ax, voxel_corners, color='green', alpha=0.3)

    # --- 探测器平面（蓝色面） ---
    det_coords = detector.coordinate  # (3, dx, dy)
    dx, dy = det_coords.shape[1], det_coords.shape[2]
    corners = [
        det_coords[:, 0, 0],
        det_coords[:, dx - 1, 0],
        det_coords[:, dx - 1, dy - 1],
        det_coords[:, 0, dy - 1]
    ]
    corners = np.array(corners)
    draw_box(ax, np.vstack([corners, corners]), color='blue', alpha=0.3)

    # --- 光源位置（红色点） ---
    src_pos = source.coordinate.reshape(3)
    ax.scatter(*src_pos, color='red', s=40, label='Source')

    # --- 探测器中心点（蓝色点） ---
    det_center = np.mean(det_coords.reshape(3, -1), axis=1)
    ax.scatter(*det_center, color='blue', s=40, label='Detector center')

    # --- 累计轨迹线 ---
    if len(source_traj) > 1:
        src_arr = np.array(source_traj)
        ax.plot(src_arr[:, 0], src_arr[:, 1], src_arr[:, 2], color='red', linestyle='-', label='Source Trajectory')

    if len(detector_center_traj) > 1:
        det_arr = np.array(detector_center_traj)
        ax.plot(det_arr[:, 0], det_arr[:, 1], det_arr[:, 2], color='blue', linestyle='--', label='Detector Trajectory')

    # 坐标轴设定
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend()
    set_axes_equal(ax) # 等比例坐标轴
    plt.pause(0.001)

