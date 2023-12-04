import numpy as np
from plyfile import PlyData
from dataclasses import dataclass

@dataclass
class GaussianDataBasic:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray
    normal: np.ndarray
    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-1]


def naive_gaussian():
    gau_xyz = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)
    gau_normal = np.array([
        1, 1, 1,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)
    gau_rot = np.array([
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0
    ]).astype(np.float32).reshape(-1, 4)
    gau_s = np.array([
        0.03, 0.03, 0.03,
        0.2, 0.03, 0.03,
        0.03, 0.2, 0.03,
        0.03, 0.03, 0.2
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = np.array([
        1, 0, 1, 
        1, 0, 0, 
        0, 1, 0, 
        0, 0, 1, 
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = (gau_c - 0.5) / 0.28209
    gau_a = np.array([
        1, 1, 1, 1
    ]).astype(np.float32).reshape(-1, 1)
    return GaussianDataBasic(
        gau_xyz,
        gau_rot,
        gau_s,
        gau_a,
        gau_c,
        gau_normal
    )


def load_ply(path):
    max_sh_degree = 3
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                    np.asarray(plydata.elements[0]["ny"]),
                    np.asarray(plydata.elements[0]["nz"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # pass activate function
    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales)
    scales = scales.astype(np.float32)
    opacities = 1/(1 + np.exp(- opacities))  # sigmoid
    opacities = opacities.astype(np.float32)
    shs = np.concatenate([features_dc.reshape(-1, 3), 
                        features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)
    return GaussianDataBasic(xyz, rots, scales, opacities, shs, normal)


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


if __name__ == "__main__":
    gs = load_ply("/Users/lmaag/Downloads/point_cloud.ply")
    a = gs.flat()
    print(a.shape)
