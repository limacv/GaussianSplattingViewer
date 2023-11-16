import numpy as np


def computeCov3D(scale,  # n, 3
                q,  # n, 4
                ):  # -> n, 3, 3
    n = len(scale)
    S = np.zeros((n, 3, 3))
    S[:, [0, 1, 2], [0, 1, 2]] = scale
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    
    R = np.stack([
		1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y),
		2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x),
		2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y)
	], axis=-1).reshape(n, 3, 3)

    M = S @ R
    Sigma = np.transpose(M, [0, 2, 1]) @ M
    return Sigma


def computeCov2D(mean_view,  # n, 3, 
                focal_x: float, 
                focal_y: float,
                cov3D,   # n, 3, 3, 
                viewmatrix):   # 3, 3
    t = mean_view
    n = len(mean_view)
    J = np.zeros((n, 3, 3))
    tz = t[:, 2]
    tz2 = tz * tz
    J[:, 0, 0] = focal_x / tz
    J[:, 2, 0] = - (focal_y * t[:, 0]) / tz2
    J[:, 1, 1] = focal_y / tz
    J[:, 2, 1] = -(focal_y * t[:, 1]) / tz2
    
    W = viewmatrix[:3, :3].T
    T = W[None] @ J

    cov = np.transpose(T, [0, 2, 1]) @ np.transpose(cov3D, [0, 2, 1]) @ T
    cov[:, 0, 0] += 0.3
    cov[:, 1, 1] += 0.3
    return cov


def pixel_coord(H, W):
    x, y = np.meshgrid(np.arange(H), np.arange(W))
    return x, y


if __name__ == "__main__":
    import glm
    # soft renderer
    state = np.load("save.npz")
    gau_xyz=state["gau_xyz"]
    gau_s=state["gau_s"]
    gau_rot=state["gau_rot"]
    gau_c=state["gau_c"]
    gau_a=state["gau_a"]
    viewmat=state["viewmat"]
    projmat=state["projmat"]
    hfovxyfocal=state["hfovxyfocal"]
    
    gau_c = np.array([
        1, 0, 1,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    ]).astype(np.float32).reshape(-1, 3)
    
    gau_xyzw = np.concatenate([gau_xyz, np.ones_like(gau_xyz[:, :1])], axis=-1)
    pos_view = viewmat[None] @ gau_xyzw[..., None]
    pos_scr = projmat[None] @ pos_view
    cov3d = computeCov3D(gau_s, gau_rot)
    cov2d = computeCov2D(pos_view[:, :3, 0], hfovxyfocal[2], hfovxyfocal[2], cov3d, viewmat)
    conic = np.linalg.inv(cov2d[:, :2, :2])
    
    w, h = 2 * hfovxyfocal[:2] * hfovxyfocal[2]
    x, y = pixel_coord(w, h)
    
    pos_scr = pos_scr[:, :3, 0] / pos_scr[:, 3:4, 0]
    gau_scr_x = (pos_scr[:, 0] + 1) / 2 * w
    gau_scr_y = (- pos_scr[:, 1] + 1) / 2 * h
    x = x[..., None] - gau_scr_x[None, None, :]
    y = gau_scr_y[None, None, :] - y[..., None]
    power = -0.5 * (conic[None, None, :, 0, 0] * x * x + conic[None, None, :, 1, 1] * y * y) - conic[None, None, :, 0, 1] * x * y
    val = np.exp(power)
    val = np.sum(val[..., None] * gau_c[None, None], axis=-2)
    
    import matplotlib.pyplot as plt
    plt.imshow(val)
    plt.show()
