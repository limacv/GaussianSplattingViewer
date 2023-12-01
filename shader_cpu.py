import numpy as np
from util_gau import computeCov2D, computeCov3D


def pixel_coord(H, W):
    x, y = np.meshgrid(np.arange(H), np.arange(W))
    return x, y


if __name__ == "__main__":
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
