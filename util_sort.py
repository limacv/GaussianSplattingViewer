import numpy as np
from util_gau import GaussianDataBasic
import open3d


def knn_normal(gaus: GaussianDataBasic):
    # could substitute normal computation on-the-fly
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(gaus.xyz)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(1)
    normal = np.asarray(pcd.normals).astype(np.float32)
    return normal


def presort_gaussian(gaus: GaussianDataBasic, normal_mod: str="depth"):
    pos = gaus.xyz
    center = np.mean(pos, axis=0)
    pos_recenter = pos - center[None]
    
    if normal_mod == "depth":
        normal = gaus.normal
    elif normal_mod == "knn":
        normal = knn_normal(gaus)
    elif normal_mod == "distance":
        normal = pos_recenter / np.linalg.norm(pos_recenter, axis=-1, keepdims=True)
    else:
        raise RuntimeError(f"Unrecognized normal mode {normal_mod}")
    
    normal_all = np.concatenate([normal, -normal])  # first len(pos) are original, second len(pos) are copys
    pos_recenter = np.concatenate([pos_recenter, pos_recenter])
    dot = (normal_all * pos_recenter).sum(axis=-1)
    index = np.argsort(dot)
    normal_all = normal_all[index]    
    index = index % len(pos)
    
    normal = np.ascontiguousarray(normal_all).astype(np.float32).reshape(-1, 3)
    index = index.astype(np.int32).reshape(-1, 1)
    return normal, index
