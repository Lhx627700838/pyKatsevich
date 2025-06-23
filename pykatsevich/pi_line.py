DEBUG = True
def find_pi_line_through_point(x, y, z, R, P, lambda_array):
    best_lambda = None
    min_error = 1e10
    print("finding pi line for:", x, y ,z)
    count = 0
    for lam_i in lambda_array:
        lam_o = lam_i + np.pi
        if lam_o > lambda_array[-1]:
            break
        
        ai = np.array([R*np.cos(lam_i), R*np.sin(lam_i), P*lam_i/(2*np.pi)])
        ao = np.array([R*np.cos(lam_o), R*np.sin(lam_o), P*lam_o/(2*np.pi)])
        
        v = ao - ai
        p = np.array([x, y, z])

        t = np.dot(p - ai, v) / np.dot(v, v)
        if 0.0 <= t <= 1.0:
            proj = ai + t * v
            dist = np.linalg.norm(proj - p)
            if dist < min_error:
                min_error = dist
                best_lambda = lam_i
                best_ai = ai
        count += 1
        if DEBUG and count%10==0:
            print("check line: ")
            print("lam_i, ai - lam_o, ao:", lam_i, ai, lam_o, ao)
            print("t for line:", t)
            print("distance between line and point:", dist)
    print("best_lambda lam_i, ai: ", best_lambda, best_ai)
    print("min error: ", min_error)
    return best_lambda #if min_error < 10.0 else None  # 用 1mm 容差限制

import numpy as np

def find_pi_line_via_rin_rout(x, y, z, R, P, D, lambda_array):
    """
    使用 rin ≈ 0 和 rout ≈ 0 方法估计穿过 (x, y, z) 的 pi-line 两端角度 lambda_in 和 lambda_out
    并返回源点位置 a_in 和 a_out，以及角度范围和补偿因子
    """
    dlambda = lambda_array[1] - lambda_array[0]

    # alpha*(λ)
    def alpha_star(lmb):
        return np.arctan2(-x*np.sin(lmb) + y*np.cos(lmb),
                          R - x*np.cos(lmb) - y*np.sin(lmb))

    alpha_vals = alpha_star(lambda_array)

    # w*(λ)
    w_star = D * np.cos(alpha_vals) / (R - x*np.cos(lambda_array) - y*np.sin(lambda_array)) * \
             (z - P/(2*np.pi)*lambda_array)

    # wtop 和 wbottom
    wtop = (D * P) / (2*np.pi*R) * ((np.pi/2 - alpha_vals) / np.cos(alpha_vals))
    wbot = -(D * P) / (2*np.pi*R) * ((np.pi/2 + alpha_vals) / np.cos(alpha_vals))

    # r_in 和 r_out
    rin = wtop - w_star
    rout = w_star - wbot

    # 找 rin > 0 且最小的项
    valid_in = np.where(rin > 0)[0]
    idx_in = valid_in[np.argmin(rin[valid_in])]
    lambda_in_near = lambda_array[idx_in]
    lambda_in = lambda_in_near + rin[idx_in] * dlambda / (rin[idx_in - 1] - rin[idx_in])
    d_in = (lambda_in_near - lambda_in) / dlambda

    # 找 rout > 0 且最小的项
    valid_out = np.where(rout > 0)[0]
    idx_out = valid_out[np.argmin(rout[valid_out])]
    lambda_out_near = lambda_array[idx_out]
    lambda_out = lambda_out_near + rout[idx_out] * dlambda / (rout[idx_out - 1] - rout[idx_out])
    d_out = (lambda_out - lambda_out_near) / dlambda

    # source 点
    a_in = np.array([R*np.cos(lambda_in), R*np.sin(lambda_in), P/(2*np.pi)*lambda_in])
    a_out = np.array([R*np.cos(lambda_out), R*np.sin(lambda_out), P/(2*np.pi)*lambda_out])
    
    # 角度范围
    angle_deg = abs(lambda_out - lambda_in) / np.pi * 180

    return {
        "lambda_in": lambda_in,
        "lambda_out": lambda_out,
        "a_in": a_in,
        "a_out": a_out,
        "angle_deg": angle_deg,
        "d_in": d_in,
        "d_out": d_out,
        "r_in": rin,
        "r_out": rout,
        "lambda_array": lambda_array
    }

if __name__ == "__main__":
    R = 610
    P = -46
    D = 1113
    lambda_array = np.linspace(0, 4*np.pi, 4032)

    x0, y0, z0 = 300, 300, -50
    result = find_pi_line_via_rin_rout(x0, y0, z0, R, P, D, lambda_array)

    print("lambda_in: ", result["lambda_in"])
    print("lambda_out:", result["lambda_out"])
    print("a_in:", result["a_in"])
    print("a_out:", result["a_out"])
    print("angle between pi-line endpoints: %.2f°" % result["angle_deg"])
    '''
    =========matlab result:
    Point [300.0 100.0 -10.0], angle = 199.85°
    approximate λ_i_near = 0.4863 rad, λ_o_near = 3.9747 rad
    λ_i = 0.4875 rad, λ_o = 3.9756 rad
    a_i = [538.94 285.73 -3.57], a_o = [-409.87 -451.78 -29.11] 
    d_in = -0.38, d_out = 0.28
    r(λ_in_near) = 0.06, r(λ_out_near) = -0.01

    Point [300.0 300.0 -50.0], angle = 133.00°
    approximate λ_i_near = 4.9474 rad, λ_o_near = 7.2699 rad
    λ_i = 4.9477 rad, λ_o = 7.2690 rad
    a_i = [142.19 -593.20 -36.22], a_o = [336.84 508.56 -53.22] 
    d_in = -0.09, d_out = -0.28
    r(λ_in_near) = 0.00, r(λ_out_near) = 0.03

    =========python result:
    lambda_in:  0.4874914715957303
    lambda_out: 3.975594271667498
    a_in: [538.94150238 285.73074215  -3.56898716]
    a_out: [-409.86964149 -451.78189094  -29.10583208]
    angle between pi-line endpoints: 199.85°

    lambda_in:  4.947655615736615
    lambda_out: 7.268975044678986
    a_in: [ 142.1923932  -593.19585578  -36.22241701]
    a_out: [336.8449743  508.56215283 -53.21709224]
    angle between pi-line endpoints: 133.00°
    '''