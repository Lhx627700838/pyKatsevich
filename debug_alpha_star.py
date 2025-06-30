import numpy as np
R0 = 610
D = 1113
x_grid = 100
y_grid = 200
z_grid = -50
z0 = 0 
P = -46

phi = -6.2317605
angle = phi/np.pi*180
print("lambda: ", angle)
d_alpha = 6.197134545454546e-04
d_w = 0.4008525174825175 
N_alpha = 1376
N_w = 144

phi *= -1 # lambda increases counterclockwise so should multiply -1
cos_phi = np.cos(phi)
sin_phi = np.sin(phi)

v_star = R0 - x_grid * cos_phi - y_grid * sin_phi
alpha_star = np.arctan((1.0 / v_star) * (-x_grid * sin_phi + y_grid * cos_phi))
w_star = D * np.cos(alpha_star) / v_star * (z_grid - z0 - P * phi / (2 * np.pi))


alpha_idx = (alpha_star/d_alpha + N_alpha/2).astype(np.int32)
w_idx = (-1*w_star/d_w + N_w/2).astype(np.int32)

print("v_star, alpha_star, w_star", v_star, alpha_star, w_star)
print("alpha_idx, w_idx", alpha_idx, w_idx)