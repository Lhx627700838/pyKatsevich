import numpy as np
from tqdm import tqdm
from pykatsevich_curve.pi_line import find_pi_line_via_rin_rout
def katsevich_backprojection_curved(
    g_filtered, lambdas, x_grid, y_grid, z_grid, R0, D, P, lambda0=0, z0=0
):
    """
    Perform backprojection for curved detector geometry (Katsevich formula, curved case).
    
    Parameters
    ----------
    g_filtered : ndarray
        Filtered projection data of shape (N_lambda, N_alpha, N_w)
    lambdas : ndarray
        1D array of projection angles λ (in radians), shape (N_lambda,)
    x_grid, y_grid, z_grid : ndarray
        3D meshgrids of the voxel coordinates
    R0 : float
        Source-to-center distance
    D : float
        Source-to-detector distance
    P : float
        Helical pitch (axial distance per full rotation)
    lambda0 : float
        Helix initial angle
    z0 : float
        Reference z position (e.g., for start of helix)
    
    Returns
    -------
    f : ndarray
        Reconstructed volume, same shape as x_grid
    """
    f = np.zeros_like(x_grid)
    d_lambda = np.abs(lambdas[1] - lambdas[0])
    N_lambda = len(lambdas)
    
    for i in tqdm(range(N_lambda), desc="Backprojecting views"):
        lam = lambdas[i]
        phi = lam + lambda0

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # v*
        v_star = R0 - x_grid * cos_phi - y_grid * sin_phi

        # α*
        alpha_star = np.arctan((1.0 / v_star) * (-x_grid * sin_phi + y_grid * cos_phi))

        # w*
        w_star = D * np.cos(alpha_star) / v_star * (z_grid - z0 - P * lam / (2 * np.pi))

        # Convert α*, w* to indices for g_filtered lookup
        # Assumption: alpha and w are uniformly sampled
        N_alpha = g_filtered.shape[1]
        N_w = g_filtered.shape[2]

        alpha_max = np.pi / 2
        w_max = D  # assume max ±D projection range

        alpha_idx = ((alpha_star + alpha_max) / (2 * alpha_max) * (N_alpha - 1)).astype(np.int32)
        w_idx = ((w_star + w_max) / (2 * w_max) * (N_w - 1)).astype(np.int32)

        # Clip indices to stay within bounds
        alpha_idx = np.clip(alpha_idx, 0, N_alpha - 1)
        w_idx = np.clip(w_idx, 0, N_w - 1)

        # Sample projection g(λ, α*, w*)
        g_sample = g_filtered[i, alpha_idx, w_idx]

        f += (g_sample / v_star) * d_lambda

    return f / (2 * np.pi)

from tqdm import tqdm
import numpy as np

def katsevich_backprojection_piline(
    g_filtered, lambdas, x_grid, y_grid, z_grid, R, D, P, lambda0, z0
):
    """
    Katsevich simplified backprojection using pi-line limited integration (weight=1).
    
    Parameters
    ----------
    g_filtered : ndarray of shape (N_lambda, N_alpha, N_w)
    lambdas : 1D array of projection angles (N_lambda,)
    x_grid, y_grid, z_grid : 3D voxel grid (shape: (Nx, Ny, Nz))
    R : source-to-center distance
    D : source-to-detector distance
    P : helical pitch
    lambda0 : helix start angle
    z0 : reference z location
    find_pi_line_via_rin_rout : function(x, y, z, R, P, D, lambda_array) -> (lambda_in, lambda_out)
    
    Returns
    -------
    f : reconstructed volume
    """
    f = np.zeros_like(x_grid)
    d_lambda = np.abs(lambdas[1] - lambdas[0])
    N_lambda = len(lambdas)
    N_alpha = g_filtered.shape[1]
    N_w = g_filtered.shape[2]
    
    alpha_max = np.pi / 2
    w_max = D  # assuming max ±D coverage

    shape = x_grid.shape
    for ix in tqdm(range(shape[0]), desc="Backprojection"):
        for iy in range(shape[1]):
            for iz in range(shape[2]):
                x = x_grid[ix, iy, iz]
                y = y_grid[ix, iy, iz]
                z = z_grid[ix, iy, iz]

                # 查找 pi-line 的 lambda 范围
                lambda_in, lambda_out = find_pi_line_via_rin_rout(x, y, z, R, P, D, lambdas)
                if lambda_in >= lambda_out:
                    continue

                # 找出落在 pi-line 范围内的 lambda 索引
                lambda_mask = (lambdas >= lambda_in) & (lambdas <= lambda_out)
                lambda_idxs = np.where(lambda_mask)[0]

                voxel_val = 0.0
                for i in lambda_idxs:
                    lam = lambdas[i]
                    phi = lam + lambda0
                    cos_phi = np.cos(phi)
                    sin_phi = np.sin(phi)

                    # v*, α*, w*
                    v_star = R - x * cos_phi - y * sin_phi
                    if v_star == 0:
                        continue  # skip divide-by-zero

                    alpha_star = np.arctan((1.0 / v_star) * (-x * sin_phi + y * cos_phi))
                    w_star = D * np.cos(alpha_star) / v_star * (z - z0 - P * lam / (2 * np.pi))

                    # 索引变换
                    alpha_idx = int(((alpha_star + alpha_max) / (2 * alpha_max)) * (N_alpha - 1))
                    w_idx = int(((w_star + w_max) / (2 * w_max)) * (N_w - 1))

                    if 0 <= alpha_idx < N_alpha and 0 <= w_idx < N_w:
                        voxel_val += g_filtered[i, alpha_idx, w_idx] / v_star

                f[ix, iy, iz] = voxel_val * d_lambda / (2 * np.pi)

    return f
