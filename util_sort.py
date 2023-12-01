import numpy as np
from util_gau import GaussianDataBasic
import open3d

def gen_normal(gaus: GaussianDataBasic):
    # currently test version
    xyz = gaus.xyz
    normal = xyz / (np.linalg.norm(xyz, axis=-1, keepdims=True) + 0.00001)
    return normal.astype(np.float32)


def presort_gaussian(gaus: GaussianDataBasic):
    pos = gaus.xyz
    center = np.mean(pos, axis=0)
    pos_recenter = pos - center[None]
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pos_recenter)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(1)
    normal = np.asarray(pcd.normals).astype(np.float32)
    
    normal_all = np.concatenate([normal, -normal])  # first len(pos) are original, second len(pos) are copys
    pos_recenter = np.concatenate([pos_recenter, pos_recenter])
    dot = (normal_all * pos_recenter).sum(axis=-1)
    index = np.argsort(dot)
    index = index % len(pos)
    
    normal = np.ascontiguousarray(normal_all).astype(np.float32).reshape(-1, 3)
    index = index.astype(np.int32).reshape(-1, 1)
    return normal, index
