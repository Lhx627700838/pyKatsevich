from pykatsevich_curve.filter import compute_hilbert_kernel, classic_hilbert_kernel
import matplotlib.pyplot as plt
import numpy as np
kernel_radius = int(1376/2)
d_alpha = 6.197134545454546e-04
H_curve_none = compute_hilbert_kernel(N=kernel_radius, d_alpha=d_alpha, apodization=None)
H_classic_none = classic_hilbert_kernel(kernel_radius, None)

H_curve_hanning = compute_hilbert_kernel(N=kernel_radius, d_alpha=d_alpha, apodization='hanning')
H_classic_hanning = classic_hilbert_kernel(kernel_radius, 'hanning')

# 横轴索引
x = np.arange(-kernel_radius, kernel_radius + 1)

# 创建并排子图
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=True)

# 左上图：curve + no window
axs[0, 0].plot(x, H_curve_none, label="Curve-derived (no window)", color='tab:blue')
axs[0, 0].set_title("compute_hilbert_kernel (no window)")
axs[0, 0].set_xlabel("t (index)")
axs[0, 0].set_ylabel("Amplitude")
axs[0, 0].grid(True)

# 右上图：classic + no window
axs[0, 1].plot(x, H_classic_none, label="Classic (no window)", color='tab:orange')
axs[0, 1].set_title("classic_hilbert_kernel (no window)")
axs[0, 1].set_xlabel("t (index)")
axs[0, 1].grid(True)

# 左下图：curve + hanning
axs[1, 0].plot(x, H_curve_hanning, label="Curve-derived (hanning)", color='tab:blue')
axs[1, 0].set_title("compute_hilbert_kernel (hanning)")
axs[1, 0].set_xlabel("t (index)")
axs[1, 0].set_ylabel("Amplitude")
axs[1, 0].grid(True)

# 右下图：classic + hanning
axs[1, 1].plot(x, H_classic_hanning, label="Classic (hanning)", color='tab:orange')
axs[1, 1].set_title("classic_hilbert_kernel (hanning)")
axs[1, 1].set_xlabel("t (index)")
axs[1, 1].grid(True)

plt.suptitle("Comparison of Hilbert Kernels", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()